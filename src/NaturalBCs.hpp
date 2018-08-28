#ifndef NATURAL_BC_HPP
#define NATURAL_BC_HPP

#include "ImplicitFunctors.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato {

  /******************************************************************************/
  /*!
    \brief Class for natural boundary conditions.
  */
    template<int SpatialDim, int DofsPerNode=SpatialDim>
    class NaturalBC
  /******************************************************************************/
  {
    const std::string    name;
    const std::string ss_name;
    Omega_h::Vector<DofsPerNode> m_flux;

  public:
  
    NaturalBC<SpatialDim,DofsPerNode>(const std::string &n, Teuchos::ParameterList &param) :
      name(n),
      ss_name(param.get<std::string>("Sides"))
      {
        auto flux = param.get<Teuchos::Array<double>>("Vector");
        for(int i=0; i<DofsPerNode; i++)
          m_flux(i) = flux[i];
      }
  
    ~NaturalBC(){}

    /*!
      \brief Get the contribution to the assembled forcing vector.
      @param aMesh Omega_h mesh that contains sidesets.
      @param aMeshSets Omega_h mesh sets that contains sideset information.
      @param forces Assembled forcing vector to which the boundary forces will be added.

      The boundary forces are integrated on the parameterized surface, \f$\phi(\xi,\psi)\f$, according to:
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
    void get( Omega_h::Mesh* aMesh, 
              const Omega_h::MeshSets& aMeshSets,
              Plato::ScalarVector& forces );

    template<typename StateScalarType,
             typename ControlScalarType,
             typename ResultScalarType>
    void get( Omega_h::Mesh* aMesh,             
              const Omega_h::MeshSets& aMeshSets,
              Plato::ScalarMultiVectorT<  StateScalarType>,
              Plato::ScalarMultiVectorT<ControlScalarType>,
              Plato::ScalarMultiVectorT< ResultScalarType> result) const;

    // ! Get sideset name
    decltype(ss_name) const& get_ss_name() const { return ss_name; }

    // ! Get the user-specified flux.
    decltype(m_flux) get_value() const { return m_flux; }

  };
  
  
  /******************************************************************************/
  /*!
    \brief Owner class that contains a vector of NaturalBC objects.
  */
  template<int SpatialDim, int DofsPerNode=SpatialDim>
  class NaturalBCs
  /******************************************************************************/
  {
  private:
    std::vector<std::shared_ptr<NaturalBC<SpatialDim,DofsPerNode>>> BCs;

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
      @param aForces Assembled forcing vector to which the boundary forces will be added.
    */
    void get( Omega_h::Mesh* aMesh,             
              const Omega_h::MeshSets& aMeshSets, 
              Plato::ScalarVector& aForces);

    template<typename StateScalarType,
             typename ControlScalarType,
             typename ResultScalarType>
    void get( Omega_h::Mesh* aMesh,             
              const Omega_h::MeshSets& aMeshSets,
              Plato::ScalarMultiVectorT<  StateScalarType>,
              Plato::ScalarMultiVectorT<ControlScalarType>,
              Plato::ScalarMultiVectorT< ResultScalarType> result) const;
  };

  /**************************************************************************/
  template<int SpatialDim, int DofsPerNode>
  template<typename StateScalarType,
           typename ControlScalarType,
           typename ResultScalarType>
  void NaturalBC<SpatialDim,DofsPerNode>::get( Omega_h::Mesh* aMesh,             
                                               const Omega_h::MeshSets& aMeshSets,
                                               Plato::ScalarMultiVectorT<  StateScalarType>,
                                               Plato::ScalarMultiVectorT<ControlScalarType>,
                                               Plato::ScalarMultiVectorT< ResultScalarType> result) const
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

    // create functor for accessing side node coordinates
    Plato::SideNodeCoordinate<SpatialDim> sideNodeCoordinate(aMesh);
    
    auto flux = m_flux;
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numFaces), LAMBDA_EXPRESSION(int iFace)
    {

      auto faceOrdinal = faceLids[iFace];

      // integrate
      //
      Omega_h::Matrix<SpatialDim, SpatialDim-1> jacobian;
      for (int d1=0; d1<SpatialDim-1; d1++)
      {
        for (int d2=0; d2<SpatialDim; d2++)
        { 
          jacobian[d1][d2] = sideNodeCoordinate(faceOrdinal,d1,d2) - sideNodeCoordinate(faceOrdinal,SpatialDim-1,d2);
        }
      }

      Scalar weight(0.0);
      if(SpatialDim==1){
        weight=1.0;
      } else
      if(SpatialDim==2){
        weight = 1.0/2.0*sqrt(jacobian[0][0]*jacobian[0][0]+jacobian[0][1]*jacobian[0][1]);
      } else
      if(SpatialDim==3){
        auto a1 = jacobian[0][1]*jacobian[1][2]-jacobian[0][2]*jacobian[1][1];
        auto a2 = jacobian[0][2]*jacobian[1][0]-jacobian[0][0]*jacobian[1][2];
        auto a3 = jacobian[0][0]*jacobian[1][1]-jacobian[0][1]*jacobian[1][0];
        weight = 1.0/6.0*sqrt(a1*a1+a2*a2+a3*a3);
      }

      int localNodeOrd[SpatialDim];
      for( int localElemOrd = face2elems_map[faceOrdinal]; 
               localElemOrd < face2elems_map[faceOrdinal+1]; ++localElemOrd ){
        auto cellOrdinal = face2elems_elems[localElemOrd];
        for( int iNode=0; iNode<nodesPerFace; iNode++){
          for( int jNode=0; jNode<nodesPerCell; jNode++){
            if( face2verts[faceOrdinal*nodesPerFace+iNode] == cell2verts[cellOrdinal*nodesPerCell + jNode] ) localNodeOrd[iNode] = jNode;
          }
        }
        for( int iNode=0; iNode<nodesPerFace; iNode++){
          for( int iDim=0; iDim<DofsPerNode; iDim++){
            auto cellDofOrdinal = localNodeOrd[iNode] * nodesPerFace + iDim;
            result(cellOrdinal,cellDofOrdinal) += weight*flux[iDim];
          }
        }
      }
    });
  }

  /****************************************************************************/
  template<int SpatialDim, int DofsPerNode>
  void NaturalBC<SpatialDim,DofsPerNode>::get( Omega_h::Mesh* aMesh,      
                                               const Omega_h::MeshSets& aMeshSets,
                                               Plato::ScalarVector& forcing )
  /****************************************************************************/
  {
    // get sideset faces
    auto& sidesets = aMeshSets[Omega_h::SIDE_SET];
    auto ssIter = sidesets.find(this->ss_name);
    auto faceLids = (ssIter->second);
    auto numFaces = faceLids.size();


    // get mesh vertices
    auto face2verts = aMesh->ask_verts_of(SpatialDim-1);

    auto numNodesPerFace = SpatialDim;

    // create functor for accessing side node coordinates
    Plato::SideNodeCoordinate<SpatialDim> sideNodeCoordinate(aMesh);
    
    auto flux = m_flux;
    auto fVec = forcing;
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numFaces), LAMBDA_EXPRESSION(int iFace)
    {

      auto faceOrdinal = faceLids[iFace];

      // integrate
      //
      Omega_h::Matrix<SpatialDim, SpatialDim-1> jacobian;
      for (int d1=0; d1<SpatialDim-1; d1++)
      {
        for (int d2=0; d2<SpatialDim; d2++)
        { 
          jacobian[d1][d2] = sideNodeCoordinate(faceOrdinal,d1,d2) - sideNodeCoordinate(faceOrdinal,SpatialDim-1,d2);
        }
      }

      Scalar weight(0.0);
      if(SpatialDim==1){
        weight=1.0;
      } else
      if(SpatialDim==2){
        weight = 1.0/2.0*sqrt(jacobian[0][0]*jacobian[0][0]+jacobian[0][1]*jacobian[0][1]);
      } else
      if(SpatialDim==3){
        auto a1 = jacobian[0][1]*jacobian[1][2]-jacobian[0][2]*jacobian[1][1];
        auto a2 = jacobian[0][2]*jacobian[1][0]-jacobian[0][0]*jacobian[1][2];
        auto a3 = jacobian[0][0]*jacobian[1][1]-jacobian[0][1]*jacobian[1][0];
        weight = 1.0/6.0*sqrt(a1*a1+a2*a2+a3*a3);
      }

      for( int iNode=0; iNode<numNodesPerFace; iNode++){
        auto nodeOrdinal = face2verts[faceOrdinal*numNodesPerFace+iNode];
        for( int iDim=0; iDim<DofsPerNode; iDim++)
          Kokkos::atomic_add(&fVec(nodeOrdinal*DofsPerNode+iDim),weight*flux[iDim]);
      } 
    });
  }


  /****************************************************************************/
  template<int SpatialDim, int DofsPerNode>
  NaturalBCs<SpatialDim,DofsPerNode>::NaturalBCs(Teuchos::ParameterList &params) : BCs()
  /****************************************************************************/
  {
    for (Teuchos::ParameterList::ConstIterator i = params.begin(); i != params.end(); ++i) {
      const Teuchos::ParameterEntry &entry = params.entry(i);
      const std::string             &name  = params.name(i);
  
      LGR_THROW_IF(!entry.isList(),
         "Parameter in Boundary Conditions block not valid.  Expect lists only.");
  
      Teuchos::ParameterList& sublist = params.sublist(name);
      const std::string type = sublist.get<std::string>("Type");
      std::shared_ptr<NaturalBC<SpatialDim,DofsPerNode>> bc;
      if ("Uniform" == type) {
        bool b_Values = sublist.isType<Teuchos::Array<double>>("Values");
        bool b_Value  = sublist.isType<double>("Value");
        if ( b_Values && b_Value ) {
          LGR_THROW_IF(true, " Natural Boundary Condition: provide EITHER 'Values' OR 'Value' Parameter.");
        } else 
        if ( b_Values ) {
          auto values = sublist.get<Teuchos::Array<double>>("Values");
          sublist.set("Vector", values);
        } else 
        if ( b_Value ) {
          Teuchos::Array<double> fluxVector(DofsPerNode, 0.0);
          std::cout << " DofsPerNode: " << DofsPerNode << std::endl;
          auto value = sublist.get<double>("Value");
          fluxVector[0] = value;
          sublist.set("Vector", fluxVector);
        } else {
          LGR_THROW_IF(true, " Natural Boundary Condition: provide either 'Values' or 'Value' Parameter.");
        }
        bc.reset(new NaturalBC<SpatialDim,DofsPerNode>(name, sublist));
      }
      else if ("Uniform Component" == type){
        Teuchos::Array<double> fluxVector(DofsPerNode, 0.0);
        auto fluxComponent = sublist.get<std::string>("Component");
        auto value = sublist.get<double>("Value");
        if( (fluxComponent == "x" || fluxComponent == "X") ) fluxVector[0] = value;
        else
        if( (fluxComponent == "y" || fluxComponent == "Y") && DofsPerNode > 1 ) fluxVector[1] = value;
        else
        if( (fluxComponent == "z" || fluxComponent == "Z") && DofsPerNode > 2 ) fluxVector[2] = value;
        sublist.set("Vector", fluxVector);
        bc.reset(new NaturalBC<SpatialDim,DofsPerNode>(name, sublist));
      } else {
        LGR_THROW_IF(true, " Natural Boundary Condition type invalid");
      }
      BCs.push_back(bc);
    }
  }

  /****************************************************************************/
  template<int SpatialDim, int DofsPerNode>
  void NaturalBCs<SpatialDim,DofsPerNode>::get( Omega_h::Mesh* aMesh,      
                                                const Omega_h::MeshSets& aMeshSets, 
                                                Plato::ScalarVector& aForcing)
  /****************************************************************************/
  {
 
    for (std::shared_ptr<NaturalBC<SpatialDim,DofsPerNode>> &bc : BCs){
      bc->get(aMesh, aMeshSets, aForcing);
    }
  }

  /**************************************************************************/
  /*!
    \brief Add the boundary load to the result workset
  */
  template<int SpatialDim, int DofsPerNode>
  template<typename StateScalarType,
           typename ControlScalarType,
           typename ResultScalarType>
  void NaturalBCs<SpatialDim,DofsPerNode>::get(Omega_h::Mesh* aMesh,      
       const Omega_h::MeshSets& aMeshSets, 
       Kokkos::View<   StateScalarType**, Kokkos::LayoutRight, Plato::MemSpace > state,
       Kokkos::View< ControlScalarType**, Kokkos::LayoutRight, Plato::MemSpace > control,
       Kokkos::View<  ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace > result) const
  /**************************************************************************/
  {
    for (const std::shared_ptr<NaturalBC<SpatialDim,DofsPerNode>> &bc : BCs){
      bc->get(aMesh, aMeshSets, state, control, result);
    }
  }


} // end namespace Plato 

#endif

