#ifndef SIMPLEX_FAD_TYPES
#define SIMPLEX_FAD_TYPES

#include <Sacado.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato {

  template<typename SimplexPhysics>
  struct SimplexFadTypes {

    using StateFad   = Sacado::Fad::SFad<Plato::Scalar,
                                         SimplexPhysics::m_numDofsPerNode*
                                         SimplexPhysics::m_numNodesPerCell>;
    using ControlFad = Sacado::Fad::SFad<Plato::Scalar,
                                         SimplexPhysics::m_numNodesPerCell>;
    using ConfigFad  = Sacado::Fad::SFad<Plato::Scalar,
                                         SimplexPhysics::m_numSpatialDims*
                                         SimplexPhysics::m_numNodesPerCell>;
  };


  // is_fad<TypesT, T>::value is true if T is of any AD type defined TypesT.
  //
  template <typename TypesT, typename T>
  struct is_fad {
    static constexpr bool value = std::is_same< T, typename TypesT::StateFad   >::value ||
                                  std::is_same< T, typename TypesT::ControlFad >::value ||
                                  std::is_same< T, typename TypesT::ConfigFad  >::value;
  };
  

  // which_fad<TypesT,T1,T2>::type returns:
  // -- compile error  if T1 and T2 are both AD types defined in TypesT,
  // -- T1             if only T1 is an AD type in TypesT,
  // -- T2             if only T2 is an AD type in TypesT,
  // -- T2             if neither are AD types.
  //
  template <typename TypesT, typename T1, typename T2>
  struct which_fad {
    static_assert( !(is_fad<TypesT,T1>::value && is_fad<TypesT,T2>::value), "Only one template argument can be an AD type.");
    using type = typename std::conditional< is_fad<TypesT,T1>::value, T1, T2 >::type;
  };
  
  
  // fad_type_t<PhysicsT,T1,T2,T3,...,TN> returns:
  // -- compile error  if more than one of T1,...,TN is an AD type in SimplexFadTypes<PhysicsT>,
  // -- type TI        if only TI is AD type in SimplexFadTypes<PhysicsT>,
  // -- TN             if none of TI are AD type in SimplexFadTypes<PhysicsT>.
  //
  template <typename TypesT, typename ...P> struct fad_type;
  template <typename TypesT, typename T> struct fad_type<TypesT, T> { using type = T; };
  template <typename TypesT, typename T, typename ...P> struct fad_type<TypesT, T, P ...> { 
    using type = typename which_fad<TypesT, T, typename fad_type<TypesT, P...>::type>::type;
  };
  template <typename PhysicsT, typename ...P> using fad_type_t = typename fad_type<SimplexFadTypes<PhysicsT>,P...>::type;



  // Create a template struct that determines the appropriate FadType:
  //
  // default: No type. Compiler error is generated for any parameter combination
  // other than the specializations below.)
  template<typename SimplexPhysics,
           typename StateScalarType, 
           typename ControlScalarType,
           typename ConfigScalarType> 
  struct WhichFad {};

  // StateFad, Scalar, Scalar:  use StateFad
  template<typename SimplexPhysics>
  struct WhichFad<
    SimplexPhysics, 
    typename SimplexFadTypes<SimplexPhysics>::StateFad, 
    Plato::Scalar,
    Plato::Scalar>
    { using scalar_type = typename SimplexFadTypes<SimplexPhysics>::StateFad; };

  // Scalar, ControlFad, Scalar: use ControlFad
  template<typename SimplexPhysics> 
  struct WhichFad<
    SimplexPhysics, 
    Plato::Scalar,
    typename SimplexFadTypes<SimplexPhysics>::ControlFad,
    Plato::Scalar>
    { using scalar_type = typename SimplexFadTypes<SimplexPhysics>::ControlFad; };

  // Scalar, Scalar, ConfigFad: use ConfigFad
  template<typename SimplexPhysics> 
  struct WhichFad<
    SimplexPhysics, 
    Plato::Scalar,
    Plato::Scalar,
    typename SimplexFadTypes<SimplexPhysics>::ConfigFad> 
    { using scalar_type = typename SimplexFadTypes<SimplexPhysics>::ConfigFad; };

  // Scalar, Scalar, Scalar: use Scalar
  template<typename SimplexPhysics> 
  struct WhichFad<
    SimplexPhysics, 
    Plato::Scalar,
    Plato::Scalar,
    Plato::Scalar>
    { using scalar_type = Plato::Scalar; };


template <typename SimplexPhysicsT>
struct EvaluationTypes
{
    static constexpr int NumControls = SimplexPhysicsT::m_numControl;
    static constexpr int SpatialDim = SimplexPhysicsT::m_numSpatialDims;
};

template <typename SimplexPhysicsT>
struct ResidualTypes : EvaluationTypes<SimplexPhysicsT>
{
  using StateScalarType   = Plato::Scalar;
  using ControlScalarType = Plato::Scalar;
  using ConfigScalarType  = Plato::Scalar;
  using ResultScalarType  = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct JacobianTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using StateScalarType   = SFadType;
  using ControlScalarType = Plato::Scalar;
  using ConfigScalarType  = Plato::Scalar;
  using ResultScalarType  = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientXTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ConfigFad;

  using StateScalarType   = Plato::Scalar;
  using ControlScalarType = Plato::Scalar;
  using ConfigScalarType  = SFadType;
  using ResultScalarType  = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientZTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ControlFad;

  using StateScalarType   = Plato::Scalar;
  using ControlScalarType = SFadType;
  using ConfigScalarType  = Plato::Scalar;
  using ResultScalarType  = SFadType;
};


template <typename SimplexPhysicsT>
struct Evaluation {
   using Residual  = ResidualTypes<SimplexPhysicsT>;
   using Jacobian  = JacobianTypes<SimplexPhysicsT>;
   using GradientZ = GradientZTypes<SimplexPhysicsT>;
   using GradientX = GradientXTypes<SimplexPhysicsT>;
};
  

}

#endif
