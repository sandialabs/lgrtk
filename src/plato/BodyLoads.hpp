#ifndef BODYLOADS_HPP
#define BODYLOADS_HPP

#include <Omega_h_expr.hpp>
#include <Omega_h_mesh.hpp>

#include <Teuchos_ParameterList.hpp>

#include "plato/alg/Basis.hpp"
#include "plato/alg/Cubature.hpp"
#include "ImplicitFunctors.hpp"

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
template <Plato::OrdinalType SpaceDim>
void 
getFunctionValues(Kokkos::View<Plato::Scalar***, Kokkos::LayoutRight, Plato::MemSpace> aQuadraturePoints,
                  const std::string& aFuncString,
                  Omega_h::Reals& aFxnValues)
/******************************************************************************/
{
  Plato::OrdinalType numCells  = aQuadraturePoints.extent(0);
  Plato::OrdinalType numPoints = aQuadraturePoints.extent(1);
  
  auto x_coords = Plato::getArray_Omega_h<Plato::Scalar>("forcing function x coords", numCells*numPoints);
  auto y_coords = Plato::getArray_Omega_h<Plato::Scalar>("forcing function y coords", numCells*numPoints);
  auto z_coords = Plato::getArray_Omega_h<Plato::Scalar>("forcing function z coords", numCells*numPoints);
  
  Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
  {
    Plato::OrdinalType entryOffset = aCellOrdinal * numPoints;
    for (Plato::OrdinalType ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
    {
      if (SpaceDim > 0) x_coords[entryOffset+ptOrdinal] = aQuadraturePoints(aCellOrdinal,ptOrdinal,0);
      if (SpaceDim > 1) y_coords[entryOffset+ptOrdinal] = aQuadraturePoints(aCellOrdinal,ptOrdinal,1);
      if (SpaceDim > 2) z_coords[entryOffset+ptOrdinal] = aQuadraturePoints(aCellOrdinal,ptOrdinal,2);
    }
  },"fill coords");
  
  Omega_h::ExprReader reader(numCells*numPoints, SpaceDim);
  if (SpaceDim > 0) reader.register_variable("x", Omega_h::any(Omega_h::Reals(x_coords)));
  if (SpaceDim > 1) reader.register_variable("y", Omega_h::any(Omega_h::Reals(y_coords)));
  if (SpaceDim > 2) reader.register_variable("z", Omega_h::any(Omega_h::Reals(z_coords)));
  
  auto result = reader.read_string(aFuncString, "Integrand");
  reader.repeat(result);
  aFxnValues = Omega_h::any_cast<Omega_h::Reals>(result);
}
  

/******************************************************************************/
template <Plato::OrdinalType SpaceDim>
void 
mapPoints(
  Omega_h::Mesh& mesh,
  Kokkos::View< Plato::Scalar**, Kokkos::LayoutRight, Plato::MemSpace> refPoints,
  Kokkos::View< Plato::Scalar***, Kokkos::LayoutRight, Plato::MemSpace> mappedPoints)
/******************************************************************************/
{
  Plato::OrdinalType numCells  = mesh.nelems();
  Plato::OrdinalType numPoints = mappedPoints.extent(1);

  Kokkos::deep_copy(mappedPoints, Plato::Scalar(0.0)); // initialize to 0

  Plato::NodeCoordinate<SpaceDim> nodeCoordinate(&mesh);

  Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal) {
    for (Plato::OrdinalType ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
    {
      Plato::OrdinalType nodeOrdinal;
      Scalar finalNodeValue = 1.0;
      for (nodeOrdinal=0; nodeOrdinal<SpaceDim; nodeOrdinal++)
      {
        Scalar nodeValue = refPoints(ptOrdinal,nodeOrdinal);
        finalNodeValue -= nodeValue;
        for (Plato::OrdinalType d=0; d<SpaceDim; d++)
        {
          mappedPoints(cellOrdinal,ptOrdinal,d) += nodeValue * nodeCoordinate(cellOrdinal,nodeOrdinal,d);
        }
      }
      nodeOrdinal = SpaceDim;
      for (Plato::OrdinalType d=0; d<SpaceDim; d++)
      {
        mappedPoints(cellOrdinal,ptOrdinal,d) += finalNodeValue * nodeCoordinate(cellOrdinal,nodeOrdinal,d);
      }
    }
  });
}


/******************************************************************************/
/*!
  \brief Class for essential boundary conditions.
*/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumDofsPerNode=SpaceDim>
class BodyLoad
/******************************************************************************/
{
  protected:
    const std::string    mName;
    const Plato::OrdinalType            mDof;
    const std::string    mFuncString;
  
  public:
  
  /**************************************************************************/
  BodyLoad<SpaceDim,NumDofsPerNode>(const std::string &n, Teuchos::ParameterList &param) :
    mName(n),
    mDof(param.get<Plato::OrdinalType>("Index")),
    mFuncString(param.get<std::string>("Function")) {}
  /**************************************************************************/
  
    ~BodyLoad(){}
  
  /**************************************************************************/
  template<typename StateScalarType, 
           typename ControlScalarType,
           typename ResultScalarType>
  void get( Omega_h::Mesh& mesh, 
       Kokkos::View<   StateScalarType**, Kokkos::LayoutRight, Plato::MemSpace >,
       Kokkos::View< ControlScalarType**, Kokkos::LayoutRight, Plato::MemSpace >,
       Kokkos::View<  ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace > result) const
  /**************************************************************************/
  {

  // get refCellQuadraturePoints, quadratureWeights
  //
  Plato::OrdinalType quadratureDegree = 1;

  Plato::OrdinalType numPoints = Plato::Cubature::getNumCubaturePoints(SpaceDim, quadratureDegree);

  Kokkos::View<Plato::Scalar**, Kokkos::LayoutRight, Plato::MemSpace>
    refCellQuadraturePoints("ref quadrature points", numPoints, SpaceDim);
  Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace>
    quadratureWeights("quadrature weights", numPoints);
  
  Plato::Cubature::getCubature(SpaceDim, quadratureDegree, refCellQuadraturePoints, quadratureWeights);


  // get basis values
  //
  Plato::Basis basis(SpaceDim);
  Plato::OrdinalType numFields = basis.basisCardinality();
  Kokkos::View<Plato::Scalar**, Kokkos::LayoutRight, Plato::MemSpace>
    refCellBasisValues("ref basis values", numFields, numPoints);

  basis.getValues(refCellQuadraturePoints, refCellBasisValues);


  // map points to physical space
  //
  Plato::OrdinalType numCells  = mesh.nelems();
  Kokkos::View<Plato::Scalar***, Kokkos::LayoutRight, Plato::MemSpace>
    quadraturePoints("quadrature points", numCells, numPoints, SpaceDim);

  mapPoints<SpaceDim>(mesh, refCellQuadraturePoints, quadraturePoints);

  
  // get integrand values at quadrature points
  //
  Omega_h::Reals fxnValues;
  getFunctionValues<SpaceDim>( quadraturePoints, mFuncString, fxnValues );

 
  // integrate and assemble
  // 
  auto dof = mDof;
  Plato::JacobianDet<SpaceDim> jacobianDet(&mesh);
  Plato::VectorEntryOrdinal<SpaceDim,SpaceDim> vectorEntryOrdinal(&mesh);
  Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
  {
    Scalar jdet = fabs(jacobianDet(cellOrdinal));

    Plato::OrdinalType entryOffset = cellOrdinal * numPoints;
    
    for (Plato::OrdinalType ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
    {
      Scalar fxnValue = fxnValues[entryOffset + ptOrdinal];
      
      Scalar weight = quadratureWeights(ptOrdinal) * jdet;
      for (Plato::OrdinalType fieldOrdinal=0; fieldOrdinal<numFields; fieldOrdinal++)
      {
        result(cellOrdinal,fieldOrdinal*NumDofsPerNode+dof) -= weight * fxnValue * refCellBasisValues(fieldOrdinal,ptOrdinal);
      }
    }
  },"assemble RHS");
  }

  /****************************************************************************/
  /*!
    \brief Add the body load to the forcing function.
    @param mesh Omega_h mesh that contains the constrained nodeset.
    @param forcing Global forcing vector to which to add the body load.
  */
  void get( Omega_h::Mesh&     mesh,
            Plato::ScalarVector&      forcing)
  /****************************************************************************/
  {

  // get refCellQuadraturePoints, quadratureWeights
  //
  Plato::OrdinalType quadratureDegree = 1;

  Plato::OrdinalType numPoints = Plato::Cubature::getNumCubaturePoints(SpaceDim,quadratureDegree);

  Kokkos::View<Plato::Scalar**, Kokkos::LayoutRight, Plato::MemSpace>
    refCellQuadraturePoints("ref quadrature points", numPoints, SpaceDim);
  Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace>
    quadratureWeights("quadrature weights", numPoints);
  
  Plato::Cubature::getCubature(SpaceDim, quadratureDegree, refCellQuadraturePoints, quadratureWeights);


  // get basis values
  //
  Plato::Basis basis(SpaceDim);
  Plato::OrdinalType numFields = basis.basisCardinality();
  Kokkos::View<Scalar**, Kokkos::LayoutRight, MemSpace>    
    refCellBasisValues("ref basis values", numFields, numPoints);

  basis.getValues(refCellQuadraturePoints, refCellBasisValues);


  // map points to physical space
  //
  Plato::OrdinalType numCells  = mesh.nelems();
  Kokkos::View<Scalar***, Kokkos::LayoutRight, MemSpace>   
    quadraturePoints("quadrature points", numCells, numPoints, SpaceDim);

  mapPoints<SpaceDim>(mesh, refCellQuadraturePoints, quadraturePoints);

  
  // get integrand values at quadrature points
  //
  Omega_h::Reals fxnValues;
  getFunctionValues<SpaceDim>( quadraturePoints, mFuncString, fxnValues );

 
  // integrate and assemble
  // 
  auto rhs = forcing;
  auto dof = mDof;
  Plato::JacobianDet<SpaceDim> jacobianDet(&mesh);
  Plato::VectorEntryOrdinal<SpaceDim,SpaceDim> vectorEntryOrdinal(&mesh);
  Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
  {
      Plato::Scalar jdet = fabs(jacobianDet(cellOrdinal));

      Plato::OrdinalType entryOffset = cellOrdinal * numPoints;
    
      for (Plato::OrdinalType ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
          Plato::Scalar fxnValue = fxnValues[entryOffset + ptOrdinal];
      
          Plato::Scalar weight = quadratureWeights(ptOrdinal) * jdet;
          for (Plato::OrdinalType fieldOrdinal=0; fieldOrdinal<numFields; fieldOrdinal++)
          {
              Plato::OrdinalType localOrdinal = vectorEntryOrdinal(cellOrdinal, fieldOrdinal, dof);
              auto contribution = -weight * fxnValue * refCellBasisValues(fieldOrdinal,ptOrdinal);
              Kokkos::atomic_add(&rhs(localOrdinal), contribution);
          }
      }
  },"assemble RHS");
}

}; // end class BodyLoad


/******************************************************************************/
/*!
  \brief Owner class that contains a vector of BodyLoad objects.
*/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumDofsPerNode=SpaceDim>
class BodyLoads
/******************************************************************************/
{
  private:
    std::vector<std::shared_ptr<BodyLoad<SpaceDim,NumDofsPerNode>>> BLs;
  public :

  /****************************************************************************/
  /*!
    \brief Constructor that parses and creates a vector of BodyLoad objects
    based on the ParameterList.
  */
  BodyLoads(Teuchos::ParameterList &params) : BLs()
  /****************************************************************************/
  {
    for (Teuchos::ParameterList::ConstIterator i = params.begin(); i != params.end(); ++i)
    {
        const Teuchos::ParameterEntry &entry = params.entry(i);
        const std::string             &name  = params.name(i);
  
        TEUCHOS_TEST_FOR_EXCEPTION(!entry.isList(), 
           std::logic_error,
           "Parameter in Body Loads block not valid.  Expect lists only.");
  
        Teuchos::ParameterList& sublist = params.sublist(name);
        std::shared_ptr<Plato::BodyLoad<SpaceDim,NumDofsPerNode>> bl;
        auto newBL = new Plato::BodyLoad<SpaceDim,NumDofsPerNode>(name, sublist);
        bl.reset(newBL);
        BLs.push_back(bl);
    }
  }

  /****************************************************************************/
  /*!
    \brief Add the body load to the forcing function.
    @param mesh Omega_h mesh that contains the constrained nodeset.
    @param forcing Global forcing vector to which to add the body load.
  */
  void get( Omega_h::Mesh& aMesh, Plato::ScalarVector& aForcing)
  /****************************************************************************/
  {
      for (std::shared_ptr<Plato::BodyLoad<SpaceDim,NumDofsPerNode>> & tbl : BLs) {
           tbl->get(aMesh, aForcing);
      }
  }

  /**************************************************************************/
  /*!
    \brief Add the body load to the result workset
  */
  template<typename StateScalarType, 
           typename ControlScalarType,
           typename ResultScalarType>
  void get( Omega_h::Mesh& aMesh,
            Kokkos::View<   StateScalarType**, Kokkos::LayoutRight, Plato::MemSpace > aState,
            Kokkos::View< ControlScalarType**, Kokkos::LayoutRight, Plato::MemSpace > aControl,
            Kokkos::View<  ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace > aResult) const
  /**************************************************************************/
  {
    for (const std::shared_ptr<Plato::BodyLoad<SpaceDim,NumDofsPerNode>> &bl : BLs){
        bl->get(aMesh, aState, aControl, aResult);
    }
  }
};

}

#endif
