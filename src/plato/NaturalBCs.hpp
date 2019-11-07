#ifndef NATURAL_BC_HPP
#define NATURAL_BC_HPP

#include "ImplicitFunctors.hpp"

#include "plato/PlatoStaticsTypes.hpp"

#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

namespace Plato {

  /******************************************************************************/
  /*!
    \brief Class for natural boundary conditions.
  */
    template<int SpatialDim, int NumDofs=SpatialDim, int DofsPerNode=NumDofs, int DofOffset=0>
    class NaturalBC
  /******************************************************************************/
  {
    const std::string    name;
    const std::string ss_name;
    Omega_h::Vector<NumDofs> mFlux;

  public:
  
    NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>(const std::string &n, Teuchos::ParameterList &param) :
      name(n),
      ss_name(param.get<std::string>("Sides"))
      {
        auto flux = param.get<Teuchos::Array<Plato::Scalar>>("Vector");
        for(int i=0; i<NumDofs; i++)
          mFlux(i) = flux[i];
      }
  
    ~NaturalBC(){}

    /*!
      \brief Get the contribution to the assembled forcing vector.
      @param aMesh Omega_h mesh that contains sidesets.
      @param aMeshSets Omega_h mesh sets that contains sideset information.
      @param aResult Assembled vector to which the boundary terms will be added.

      The boundary terms are integrated on the parameterized surface, \f$\phi(\xi,\psi)\f$, according to:
      \f{eqnarray*}{
          \phi(\xi,\psi)=
            \left\{ 
             \begin{array}{ccc} 
               N_I\left(\xi,\psi\right) x_I &
               N_I\left(\xi,\psi\right) y_I &
               N_I\left(\xi,\psi\right) z_I 
             \end{array} 
            \right\} \\
          f^{el}_{Ii} = \int_{\partial\Omega_{\xi}} N_I\left(\xi,\psi\right) t_i 
                \left|\left| 
                  \frac{\partial\phi}{\partial\xi} \times \frac{\partial\phi}{\partial\psi} 
                \right|\right| d\xi d\psi
      \f}
    */

    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void get( Omega_h::Mesh* aMesh,             
              const Omega_h::MeshSets& aMeshSets,
              Plato::ScalarMultiVectorT<  StateScalarType>,
              Plato::ScalarMultiVectorT<ControlScalarType>,
              Plato::ScalarArray3DT    < ConfigScalarType>,
              Plato::ScalarMultiVectorT< ResultScalarType>,
              Plato::Scalar scale) const;

    // ! Get sideset name
    decltype(ss_name) const& get_ss_name() const { return ss_name; }

    // ! Get the user-specified flux.
    decltype(mFlux) get_value() const { return mFlux; }

  };
  
  
  /******************************************************************************/
  /*!
    \brief Owner class that contains a vector of NaturalBC objects.
  */
  template<int SpatialDim, int NumDofs=SpatialDim, int DofsPerNode=NumDofs, int DofOffset=0>
  class NaturalBCs
  /******************************************************************************/
  {
  private:
    std::vector<std::shared_ptr<NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>>> BCs;

  public :

    /*!
      \brief Constructor that parses and creates a vector of NaturalBC objects
      based on the ParameterList.
    */
    NaturalBCs(Teuchos::ParameterList &params);

    /*!
      \brief Get the contribution to the assembled forcing vector from the owned boundary conditions.
      @param aMesh Omega_h mesh that contains sidesets.
      @param aMeshSets Omega_h mesh sets that contains sideset information.
      @param aResult Assembled vector to which the boundary terms will be added.
    */
    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void get( Omega_h::Mesh* aMesh,             
              const Omega_h::MeshSets& aMeshSets,
              Plato::ScalarMultiVectorT<  StateScalarType>,
              Plato::ScalarMultiVectorT<ControlScalarType>,
              Plato::ScalarArray3DT    < ConfigScalarType>,
              Plato::ScalarMultiVectorT< ResultScalarType>,
              Plato::Scalar scale = 1.0) const;
  };

  /**************************************************************************/
  template<int SpatialDim, int NumDofs, int DofsPerNode, int DofOffset>
  template<typename StateScalarType,
           typename ControlScalarType,
           typename ConfigScalarType,
           typename ResultScalarType>
  void NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>::get( Omega_h::Mesh* aMesh,             
                                               const Omega_h::MeshSets& aMeshSets,
                                               Plato::ScalarMultiVectorT<  StateScalarType> aState,
                                               Plato::ScalarMultiVectorT<ControlScalarType> aControl,
                                               Plato::ScalarArray3DT    < ConfigScalarType> aConfig,
                                               Plato::ScalarMultiVectorT< ResultScalarType> aResult,
                                               Plato::Scalar scale) const
  /**************************************************************************/
  {
    // get sideset faces
    auto& sidesets = aMeshSets[Omega_h::SIDE_SET];
    auto ssIter = sidesets.find(this->ss_name);
    auto faceLids = (ssIter->second);
    auto numFaces = faceLids.size();


    // get mesh vertices
    auto face2verts = aMesh->ask_verts_of(SpatialDim-1);
    auto cell2verts = aMesh->ask_elem_verts();

    auto face2elems = aMesh->ask_up(SpatialDim - 1, SpatialDim);
    auto face2elems_map   = face2elems.a2ab;
    auto face2elems_elems = face2elems.ab2b;

    auto nodesPerFace = SpatialDim;
    auto nodesPerCell = SpatialDim+1;

    Plato::ScalarArray3DT<ConfigScalarType> tJacobian("jacobian", numFaces, SpatialDim-1, SpatialDim);

    auto flux = mFlux;
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numFaces), LAMBDA_EXPRESSION(int iFace)
    {

      auto faceOrdinal = faceLids[iFace];

      // for each element that the face is connected to: (either 1 or 2)
      for( int localElemOrd = face2elems_map[faceOrdinal]; localElemOrd < face2elems_map[faceOrdinal+1]; ++localElemOrd ){

        // create a map from face local node index to elem local node index
        int localNodeOrd[SpatialDim];
        auto cellOrdinal = face2elems_elems[localElemOrd];
        for( int iNode=0; iNode<nodesPerFace; iNode++){
          for( int jNode=0; jNode<nodesPerCell; jNode++){
            if( face2verts[faceOrdinal*nodesPerFace+iNode] == cell2verts[cellOrdinal*nodesPerCell + jNode] ) localNodeOrd[iNode] = jNode;
          }
        }

        // compute jacobian from aConfig
        for( int iNode=0; iNode<SpatialDim-1; iNode++){
          for( int iDim=0; iDim<SpatialDim; iDim++){
            tJacobian(iFace,iNode,iDim) = aConfig(cellOrdinal, localNodeOrd[iNode], iDim)
                                        - aConfig(cellOrdinal, localNodeOrd[SpatialDim-1], iDim);
          }
        }
        ConfigScalarType weight(0.0);
        if(SpatialDim==1){
          weight=scale;
        } else
        if(SpatialDim==2){
          weight = scale/2.0*sqrt(tJacobian(iFace,0,0)*tJacobian(iFace,0,0)+tJacobian(iFace,0,1)*tJacobian(iFace,0,1));
        } else
        if(SpatialDim==3){
          auto a1 = tJacobian(iFace,0,1)*tJacobian(iFace,1,2)-tJacobian(iFace,0,2)*tJacobian(iFace,1,1);
          auto a2 = tJacobian(iFace,0,2)*tJacobian(iFace,1,0)-tJacobian(iFace,0,0)*tJacobian(iFace,1,2);
          auto a3 = tJacobian(iFace,0,0)*tJacobian(iFace,1,1)-tJacobian(iFace,0,1)*tJacobian(iFace,1,0);
          weight = scale/6.0*sqrt(a1*a1+a2*a2+a3*a3);
        }

        // project into aResult workset
        for( int iNode=0; iNode<nodesPerFace; iNode++){
          for( int iDof=0; iDof<NumDofs; iDof++){
            auto cellDofOrdinal = localNodeOrd[iNode] * DofsPerNode + iDof + DofOffset;
            aResult(cellOrdinal,cellDofOrdinal) += weight*flux[iDof];
          }
        }
      }

    });
  }

  /****************************************************************************/
  template<int SpatialDim, int NumDofs, int DofsPerNode, int DofOffset>
  NaturalBCs<SpatialDim,NumDofs,DofsPerNode,DofOffset>::NaturalBCs(Teuchos::ParameterList &params) : BCs()
  /****************************************************************************/
  {
    for (Teuchos::ParameterList::ConstIterator i = params.begin(); i != params.end(); ++i) {
      const Teuchos::ParameterEntry &entry = params.entry(i);
      const std::string             &name  = params.name(i);
  
      TEUCHOS_TEST_FOR_EXCEPTION(!entry.isList(),
         std::logic_error,
         "Parameter in Boundary Conditions block not valid.  Expect lists only.");
  
      Teuchos::ParameterList& sublist = params.sublist(name);
      const std::string type = sublist.get<std::string>("Type");
      std::shared_ptr<NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>> bc;
      if ("Uniform" == type) {
        bool b_Values = sublist.isType<Teuchos::Array<Plato::Scalar>>("Values");
        bool b_Value  = sublist.isType<Plato::Scalar>("Value");
        if ( b_Values && b_Value ) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, 
             std::logic_error,
             " Natural Boundary Condition: provide EITHER 'Values' OR 'Value' Parameter.");
        } else 
        if ( b_Values ) {
          auto values = sublist.get<Teuchos::Array<Plato::Scalar>>("Values");
          sublist.set("Vector", values);
        } else 
        if ( b_Value ) {
          Teuchos::Array<Plato::Scalar> fluxVector(NumDofs, 0.0);
          auto value = sublist.get<Plato::Scalar>("Value");
          fluxVector[0] = value;
          sublist.set("Vector", fluxVector);
        } else {
          TEUCHOS_TEST_FOR_EXCEPTION(true, 
             std::logic_error,
             " Natural Boundary Condition: provide either 'Values' or 'Value' Parameter.");
        }
        bc.reset(new NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>(name, sublist));
      }
      else if ("Uniform Component" == type){
        Teuchos::Array<Plato::Scalar> fluxVector(NumDofs, 0.0);
        auto fluxComponent = sublist.get<std::string>("Component");
        auto value = sublist.get<Plato::Scalar>("Value");
        if( (fluxComponent == "x" || fluxComponent == "X") ) fluxVector[0] = value;
        else
        if( (fluxComponent == "y" || fluxComponent == "Y") && DofsPerNode > 1 ) fluxVector[1] = value;
        else
        if( (fluxComponent == "z" || fluxComponent == "Z") && DofsPerNode > 2 ) fluxVector[2] = value;
        sublist.set("Vector", fluxVector);
        bc.reset(new NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>(name, sublist));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, 
           std::logic_error,
           " Natural Boundary Condition type invalid");
      }
      BCs.push_back(bc);
    }
  }

  /**************************************************************************/
  /*!
    \brief Add the boundary load to the result workset
  */
  template<int SpatialDim, int NumDofs, int DofsPerNode, int DofOffset>
  template<typename StateScalarType,
           typename ControlScalarType,
           typename ConfigScalarType,
           typename ResultScalarType>
  void NaturalBCs<SpatialDim,NumDofs,DofsPerNode,DofOffset>::get(Omega_h::Mesh* aMesh,      
       const Omega_h::MeshSets& aMeshSets, 
       Plato::ScalarMultiVectorT<  StateScalarType> aState,
       Plato::ScalarMultiVectorT<ControlScalarType> aControl,
       Plato::ScalarArray3DT    < ConfigScalarType> aConfig,
       Plato::ScalarMultiVectorT< ResultScalarType> aResult,
       Plato::Scalar aScale) const
  /**************************************************************************/
  {
    for (const auto &bc : BCs){
      bc->get(aMesh, aMeshSets, aState, aControl, aConfig, aResult, aScale);
    }
  }


} // end namespace Plato 

#endif

