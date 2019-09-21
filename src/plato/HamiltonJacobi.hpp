#ifndef HAMILTONJACOBI_HPP
#define HAMILTONJACOBI_HPP

#include "Kokkos_Atomic.hpp"

#include "Omega_h_build.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_mesh.hpp"

#include "ImplicitFunctors.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/PlatoStaticsTypes.hpp"

template<int SpatialDim>
KOKKOS_INLINE_FUNCTION
Plato::Scalar unitize(Omega_h::Vector<SpatialDim> & v)
{
  Plato::Scalar mag = 0.;
  for (int dim=0; dim<SpatialDim; ++dim) mag += v[dim] * v[dim];
  mag = sqrt(mag);
  for (int dim=0; dim<SpatialDim; ++dim) v[dim] /= mag;
  return mag;
}

template<int SpatialDim>
KOKKOS_INLINE_FUNCTION
Plato::Scalar length(const Omega_h::Vector<SpatialDim> & v)
{
  Plato::Scalar sqrMag = 0.;
  for (int dim=0; dim<SpatialDim; ++dim) sqrMag += v[dim] * v[dim];
  return sqrt(sqrMag);
}

template<int SpatialDim>
KOKKOS_INLINE_FUNCTION
Plato::Scalar length_squared(const Omega_h::Vector<SpatialDim> & v)
{
  Plato::Scalar sqrMag = 0.;
  for (int dim=0; dim<SpatialDim; ++dim) sqrMag += v[dim] * v[dim];
  return sqrMag;
}

template<int SpatialDim>
KOKKOS_INLINE_FUNCTION
Plato::Scalar dot(const Omega_h::Vector<SpatialDim> & v1, const Omega_h::Vector<SpatialDim> & v2)
{
  Plato::Scalar result = 0.;
  for (int dim=0; dim<SpatialDim; ++dim) result += v1[dim] * v2[dim];
  return result;
}

template<int SpatialDim>
struct ProblemFields
{
  typename Plato::ScalarVector mRHS;
  typename Plato::ScalarVector mRHSNorm;
  typename Plato::ScalarVector mElementSpeed;
  typename Plato::ScalarVector mNodalSpeed;
  typename Plato::ScalarMultiVector mLevelSet;
  typename Plato::ScalarMultiVector mLevelSetHistory;
  Plato::OrdinalType mNumTimeSteps = 0;
  Plato::OrdinalType mCurrentState = 0;
  bool mUseElementSpeed = true;
};

template<int SpatialDim>
void declare_fields(Omega_h::Mesh & aMesh, ProblemFields<SpatialDim> & aFields, const bool aUseElementSpeed = true)
{
    constexpr Plato::OrdinalType tNumStates = 2;
    const Plato::OrdinalType tElemCount = aMesh.nelems();
    const Plato::OrdinalType tNodeCount = aMesh.nverts();
    aFields.mRHS = Plato::ScalarVector("nodal RHS", tNodeCount);
    aFields.mRHSNorm = Plato::ScalarVector("nodal RHS norm", tNodeCount);
    aFields.mLevelSet = Plato::ScalarMultiVector("nodal levelSet", tNodeCount, tNumStates);
    aFields.mLevelSetHistory = Plato::ScalarMultiVector("nodal LevelSetHistory", tNodeCount, aFields.mNumTimeSteps);
    aFields.mUseElementSpeed = aUseElementSpeed;
    if (aUseElementSpeed == true)
    {
    	aFields.mElementSpeed = Plato::ScalarVector("element speed", tElemCount);
    }
    else
    {
    	aFields.mNodalSpeed = Plato::ScalarVector("nodal speed", tNodeCount);
    }
}

template<int SpatialDim>
void initialize_constant_speed(Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields, const Plato::Scalar speed)
{
  if (fields.mUseElementSpeed)
  {
	  auto elementSpeed = fields.mElementSpeed;
	  auto f = LAMBDA_EXPRESSION(int elem) {
		elementSpeed(elem) = speed;
	  };
	  Kokkos::parallel_for(omega_h_mesh.nelems(), f);
  }
  else
  {
	  auto nodalSpeed = fields.mNodalSpeed;
	  auto f = LAMBDA_EXPRESSION(int node) {
		nodalSpeed(node) = speed;
	  };
	  Kokkos::parallel_for(omega_h_mesh.nverts(), f);
  }
}

KOKKOS_INLINE_FUNCTION
Plato::Scalar Heaviside(const Plato::Scalar signedDist, const Plato::Scalar eps)
{
  if (signedDist < -eps) return 0.;
  else if (signedDist > eps) return 1.;

  constexpr Plato::Scalar pi = 3.1415926535897932385;
  return 0.5*(1+ signedDist/eps + sin(pi*signedDist/eps)/pi);
}

KOKKOS_INLINE_FUNCTION
Plato::Scalar Delta(const Plato::Scalar signedDist, const Plato::Scalar eps)
{
  if (signedDist < -eps || signedDist > eps) return 0.;

  constexpr Plato::Scalar pi = 3.1415926535897932385;
  return 0.5/eps * (1. + cos(pi*signedDist/eps));
}

template<int SpatialDim, class Lambda>
Plato::Scalar level_set_integral(Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields, Plato::Scalar eps, const Lambda & levelSetFn, Plato::Scalar levelSet)
{
  constexpr int nodesPerElem = SpatialDim + 1;
  Plato::NodeCoordinate<SpatialDim> nodeCoordinate(&omega_h_mesh);
  Plato::ComputeVolume<SpatialDim> computeElemVolume(nodeCoordinate);
  auto levelSetField = fields.mLevelSet;
  auto elems2Verts = omega_h_mesh.ask_elem_verts();

  auto computeIntegral = LAMBDA_EXPRESSION(int elem, Plato::Scalar & integral) {
    Plato::Scalar elemVolume = computeElemVolume(elem);

    Plato::Scalar avgLS = 0.;
    for (int n=0; n<nodesPerElem; ++n)
    {
      avgLS += levelSetField(elems2Verts[elem * nodesPerElem + n],fields.mCurrentState)/nodesPerElem;
    }

    integral += elemVolume * levelSetFn(avgLS - levelSet, eps);
  };
  Plato::Scalar integral = 0.;
  Kokkos::parallel_reduce(omega_h_mesh.nelems(), computeIntegral, integral);
  return integral;
}

template<int SpatialDim>
Plato::Scalar level_set_volume_rate_of_change(Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields, Plato::Scalar eps, Plato::Scalar levelSet = 0.)
{
  constexpr int nodesPerElem = SpatialDim + 1;
  Plato::NodeCoordinate<SpatialDim> nodeCoordinate(&omega_h_mesh);
  Plato::ComputeVolume<SpatialDim> computeElemVolume(nodeCoordinate);
  auto levelSetField = fields.mLevelSet;
  auto elems2Verts = omega_h_mesh.ask_elem_verts();

  auto computeIntegral = LAMBDA_EXPRESSION(int elem, Plato::Scalar & integral) {
    Plato::Scalar elemVolume = computeElemVolume(elem);

    Plato::Scalar avgLS = 0.;
    for (int n=0; n<nodesPerElem; ++n)
    {
      avgLS += levelSetField(elems2Verts[elem * nodesPerElem + n],fields.mCurrentState)/nodesPerElem;
    }

    Plato::Scalar elemSpeed = 0.;
    if (fields.mUseElementSpeed)
    {
      elemSpeed = fields.mElementSpeed[elem];
    }
    else
    {
      for (int n=0; n<nodesPerElem; ++n)
        elemSpeed += fields.mNodalSpeed(elems2Verts[elem * nodesPerElem + n])/nodesPerElem;
    }

    integral += elemVolume * elemSpeed * Delta(avgLS - levelSet, eps);
  };
  Plato::Scalar integral = 0.;
  Kokkos::parallel_reduce(omega_h_mesh.nelems(), computeIntegral, integral);
  return integral;
}

template<int SpatialDim>
Plato::Scalar level_set_volume(Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields, Plato::Scalar eps, Plato::Scalar levelSet = 0.)
{
  auto fn = LAMBDA_EXPRESSION(const Plato::Scalar signedDist, const Plato::Scalar eps) {
    return Heaviside(signedDist, eps);
  };
  return level_set_integral(omega_h_mesh, fields, eps, fn, levelSet);
}

template<int SpatialDim>
Plato::Scalar level_set_area(Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields, Plato::Scalar eps, Plato::Scalar levelSet = 0.)
{
  auto fn = LAMBDA_EXPRESSION(const Plato::Scalar signedDist, const Plato::Scalar eps) {
    return Delta(signedDist, eps);
  };
  return level_set_integral(omega_h_mesh, fields, eps, fn, levelSet);
}

template<int SpatialDim>
Plato::Scalar mesh_minimum_length_scale(Omega_h::Mesh & omega_h_mesh)
{
  constexpr int nodesPerCell = SpatialDim + 1;
  Plato::NodeCoordinate<SpatialDim> nodeCoordinate(&omega_h_mesh);
  Plato::ComputeGradient<SpatialDim> computeGradient(nodeCoordinate);

  auto findMaxGradSqrMagnitude = LAMBDA_EXPRESSION(int e, Plato::Scalar & maxGradSqrMag) {
    Plato::Scalar elemVolume;
    Omega_h::Vector<SpatialDim> gradients[nodesPerCell];
    computeGradient(e, gradients, elemVolume);

    for (unsigned n=0; n<nodesPerCell; ++n)
    {
      const Plato::Scalar gradSqrMag = length_squared(gradients[n]);
      maxGradSqrMag = Omega_h::max2(maxGradSqrMag, gradSqrMag);
    }
  };

  Plato::Scalar maxGradSqrMag ;
  Kokkos::Max<Plato::Scalar> xreducer(maxGradSqrMag);
  Kokkos::parallel_reduce(omega_h_mesh.nelems(), findMaxGradSqrMagnitude, xreducer);

  return 1./sqrt(maxGradSqrMag);
}

template<typename TYPE, int SIZE>
struct SumArray
{
  KOKKOS_INLINE_FUNCTION
  SumArray() { for (int i=0; i<SIZE; ++i) data[i] = 0; }

  KOKKOS_INLINE_FUNCTION
  TYPE & operator[](int i) { return data[i]; }

  KOKKOS_INLINE_FUNCTION
  volatile SumArray<TYPE,SIZE> & operator+=(const volatile SumArray<TYPE,SIZE> & RHS) volatile
  {
    for (int i=0; i<SIZE; ++i) data[i] += RHS.data[i];
    return *this;
  }

  TYPE data[SIZE];
};

template<int SpatialDim>
bool domain_contains_interface(Omega_h::Mesh & aMeshOmegaH, ProblemFields<SpatialDim> & aFields)
{
  auto tLevelSet = aFields.mLevelSet;

  auto tFunctor = LAMBDA_EXPRESSION(int aIndex, SumArray<size_t,2> & aNumPosNeg)
  {
      size_t tValue = tLevelSet(aIndex, aFields.mCurrentState) < 0. ? 1 : 0;
      aNumPosNeg[0] += tValue;
      tValue = tLevelSet(aIndex, aFields.mCurrentState) > 0. ? 1 : 0;
      aNumPosNeg[1] += tValue;
  };

  SumArray<size_t, 2> tNumPosNeg;
  Kokkos::parallel_reduce(aMeshOmegaH.nverts(), tFunctor, tNumPosNeg);

  bool tOutput = tNumPosNeg[0] > 0 && tNumPosNeg[1] > 0;
  return tOutput;
}

/******************************************************************************//**
 * @brief Initialize level set field
 * @param [in] aOmega_h_Mesh mesh database
 * @param [in/out] aFields problem fields data structure
 * @param [in] aLevelSetFunction level set function interface
 * @param [in] aMultiplier level set field multiplier (default = 1)
**********************************************************************************/
template<int SpatialDim, class Lambda>
inline void initialize_level_set(Omega_h::Mesh & aOmega_h_Mesh,
                                 ProblemFields<SpatialDim> & aFields,
                                 const Lambda & aLevelSetFunction,
                                 const Plato::Scalar aMultiplier = 1.0)
{
    auto tLevelSet = aFields.mLevelSet;
    const Omega_h::Reals tCoords = aOmega_h_Mesh.coords();
    auto tLambdaExp = LAMBDA_EXPRESSION(int tIndex)
    {
        const Plato::Scalar tX = tCoords[tIndex*SpatialDim + 0];
        const Plato::Scalar tY = tCoords[tIndex*SpatialDim + 1];
        const Plato::Scalar tZ = (SpatialDim > 2) ? tCoords[tIndex*SpatialDim + 2] : 0.0;
        tLevelSet(tIndex, aFields.mCurrentState) = aLevelSetFunction(tX,tY,tZ) * aMultiplier;
    };
    Kokkos::parallel_for(aOmega_h_Mesh.nverts(), tLambdaExp);
}
// function initialize_level_set

template<int SpatialDim>
Plato::Scalar max_element_speed(Omega_h::Mesh & omega_h_mesh,
		ProblemFields<SpatialDim> & aFields)
{
  auto elementSpeed = aFields.mElementSpeed;
  auto findElemMaxSpeed = LAMBDA_EXPRESSION(int e, Plato::Scalar & maxSpeed) {
    maxSpeed = Omega_h::max2(maxSpeed, elementSpeed(e));
  };

  Plato::Scalar maxSpeed ;
  Kokkos::Max<Plato::Scalar> xreducer(maxSpeed);
  Kokkos::parallel_reduce(omega_h_mesh.nelems(), findElemMaxSpeed, xreducer);

  return maxSpeed;
}

template<int SpatialDim>
Plato::Scalar max_nodal_speed(Omega_h::Mesh & omega_h_mesh,
		ProblemFields<SpatialDim> & aFields)
{
  auto nodalSpeed = aFields.mNodalSpeed;
  auto findNodalMaxSpeed = LAMBDA_EXPRESSION(int n, Plato::Scalar & maxSpeed) {
    maxSpeed = Omega_h::max2(maxSpeed, nodalSpeed(n));
  };

  Plato::Scalar maxSpeed ;
  Kokkos::Max<Plato::Scalar> xreducer(maxSpeed);
  Kokkos::parallel_reduce(omega_h_mesh.nverts(), findNodalMaxSpeed, xreducer);

  return maxSpeed;
}

inline void scale_field(const Plato::Scalar scale,
		Omega_h::Mesh & omega_h_mesh,
		Plato::ScalarVector & field,
		const Plato::OrdinalType count)
{
  auto scaleField = LAMBDA_EXPRESSION(int n) {
    field(n) *= scale;
  };
  Kokkos::parallel_for(count, scaleField);
}

/******************************************************************************//**
 * @brief Initialize nodal speed field
 * @param [in] aOmega_h_Mesh mesh database
 * @param [in/out] aFields problem fields data structure
 * @param [in] aSpeedFunction level set interface speed interface
**********************************************************************************/
template<int SpatialDim, class Lambda>
inline void initialize_nodal_speed(Omega_h::Mesh & aOmega_h_Mesh,
                                 ProblemFields<SpatialDim> & aFields,
                                 const Lambda & aSpeedFunction)
{
    auto nodalSpeed = aFields.mNodalSpeed;
    const Omega_h::Reals tCoords = aOmega_h_Mesh.coords();
    auto tLambdaExp = LAMBDA_EXPRESSION(int tNode)
    {
        const Plato::Scalar tX = tCoords[tNode*SpatialDim + 0];
        const Plato::Scalar tY = tCoords[tNode*SpatialDim + 1];
        const Plato::Scalar tZ = (SpatialDim > 2) ? tCoords[tNode*SpatialDim + 2] : 0.0;
        nodalSpeed(tNode) = aSpeedFunction(tX,tY,tZ);
    };
    Kokkos::parallel_for(aOmega_h_Mesh.nverts(), tLambdaExp);
}
// function initialize_nodal_speed

/******************************************************************************//**
 * @brief Initialize element speed field
 * @param [in] aOmega_h_Mesh mesh database
 * @param [in/out] aFields problem fields data structure
 * @param [in] aSpeedFunction level set interface speed interface
**********************************************************************************/
template<int SpatialDim, class Lambda>
inline void initialize_element_speed(Omega_h::Mesh & aOmega_h_Mesh,
                                     ProblemFields<SpatialDim> & aFields,
                                     const Lambda & aSpeedFunction)
{
    constexpr int nodesPerElem = SpatialDim + 1;
    constexpr Plato::Scalar invNPE = 1. / nodesPerElem;

    auto elementSpeed = aFields.mElementSpeed;
    auto elems2Verts = aOmega_h_Mesh.ask_elem_verts();
    const Omega_h::Reals tCoords = aOmega_h_Mesh.coords();
    auto tLambdaExp = LAMBDA_EXPRESSION(int tElem)
    {
        Plato::Scalar tX = 0.;
        Plato::Scalar tY = 0.;
        Plato::Scalar tZ = 0.;
        for (unsigned n=0; n<nodesPerElem; ++n)
        {
            auto node = elems2Verts[tElem * nodesPerElem + n];
            tX += invNPE*tCoords[node*SpatialDim+0];
            tY += invNPE*tCoords[node*SpatialDim+1];
            tZ += (SpatialDim > 2) ? invNPE*tCoords[node*SpatialDim+2] : 0.0;
        }
        elementSpeed(tElem) = aSpeedFunction(tX,tY,tZ);
    };
    Kokkos::parallel_for(aOmega_h_Mesh.nelems(), tLambdaExp);
}
// function initialize_element_speed

/******************************************************************************//**
 * @brief Initialize speed field
 * @param [in] aOmega_h_Mesh mesh database
 * @param [in/out] aFields problem fields data structure
 * @param [in] aSpeedFunction level set interface speed interface
**********************************************************************************/
template<int SpatialDim, class Lambda>
inline void initialize_interface_speed(Omega_h::Mesh & aOmega_h_Mesh,
                                 ProblemFields<SpatialDim> & aFields,
                                 const Lambda & aSpeedFunction)
{
      if (aFields.mUseElementSpeed)
      {
    	  initialize_element_speed(aOmega_h_Mesh, aFields, aSpeedFunction);
      }
      else
      {
          initialize_nodal_speed(aOmega_h_Mesh, aFields, aSpeedFunction);
      }
}
// function initialize_level_set

/******************************************************************************//**
 * @brief normalize speed field
 * @param [in] aOmega_h_Mesh mesh database
 * @param [in/out] aFields problem fields data structure
 * @param [out] tMaxSpeed maximum interface speed
**********************************************************************************/
template<int SpatialDim>
inline Plato::Scalar normalize_interface_speed(Omega_h::Mesh & aOmega_h_Mesh,
                                 ProblemFields<SpatialDim> & aFields)
{
	  Plato::Scalar max_speed = (aFields.mUseElementSpeed) ?
			  max_element_speed(aOmega_h_Mesh, aFields) :
			  max_nodal_speed(aOmega_h_Mesh, aFields);

      if (aFields.mUseElementSpeed)
      {
    	  scale_field(1./max_speed, aOmega_h_Mesh, aFields.mElementSpeed, aOmega_h_Mesh.nelems());
      }
      else
      {
          scale_field(1./max_speed, aOmega_h_Mesh, aFields.mNodalSpeed, aOmega_h_Mesh.nverts());
      }

      return max_speed;
}
// function initialize_level_set

/******************************************************************************//**
 * @brief Offset level set field
 * @param [in] aOmega_h_Mesh mesh database
 * @param [in/out] aFields problem fields data structure
 * @param [in] aOffset level set offset rate
**********************************************************************************/
template<int SpatialDim>
inline void offset_level_set(Omega_h::Mesh & aOmega_h_Mesh, ProblemFields<SpatialDim> & aFields, const Plato::Scalar & aOffset)
{
    auto tLevelSet = aFields.mLevelSet;
    auto tLambdaExp = LAMBDA_EXPRESSION(int tIndex)
    {
        tLevelSet(tIndex, aFields.mCurrentState) += aOffset;
    };
    Kokkos::parallel_for(aOmega_h_Mesh.nverts(), tLambdaExp);
}
// function offset_level_set

template<int SpatialDim>
class AssembleElementSpeedHamiltonian
{
  // Implementation of Barth-Sethian positive coefficient scheme for element speed fields.
private:
  const Omega_h::LOs mElems2Verts;
  const Plato::NodeCoordinate<SpatialDim> mNodeCoordinate;
  const Plato::ComputeGradient<SpatialDim> mComputeGradient;
  const ProblemFields<SpatialDim> mFields;
  const Plato::Scalar mEps;

  DEVICE_TYPE inline
  Plato::Scalar get_element_speed(const ProblemFields<SpatialDim> & fields, int elem) const
  {
	if (fields.mUseElementSpeed) return fields.mElementSpeed[elem];

	constexpr int nodesPerElem = SpatialDim + 1;
	Plato::Scalar elementSpeed = 0.;
	for (int n=0; n<nodesPerElem; ++n)
		elementSpeed += mFields.mNodalSpeed(mElems2Verts[elem * nodesPerElem + n]);
    return elementSpeed/nodesPerElem;
  }
public:
  AssembleElementSpeedHamiltonian(Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields, const Plato::Scalar eps) :
    mElems2Verts(omega_h_mesh.ask_elem_verts()),
    mNodeCoordinate(&omega_h_mesh),
    mComputeGradient(mNodeCoordinate),
    mFields(fields),
    mEps(eps) {}

  DEVICE_TYPE inline
  void operator()(int elem) const
  {
    constexpr int nodesPerElem = SpatialDim + 1;
    Plato::Scalar elemVolume;
    Omega_h::Vector<SpatialDim> gradients[nodesPerElem];
    mComputeGradient(elem, gradients, elemVolume);

    Omega_h::Vector<SpatialDim> normalDir;
    for (int dim=0; dim<SpatialDim; ++dim) normalDir[dim] = 0.;
    for (int n=0; n<nodesPerElem; ++n)
    {
      const Plato::Scalar LSOld = mFields.mLevelSet(mElems2Verts[elem * nodesPerElem + n], mFields.mCurrentState);
      for (int dim=0; dim<SpatialDim; ++dim) normalDir[dim] += gradients[n][dim] * LSOld;
    }
    unitize(normalDir);

    const Plato::Scalar elementSpeed = get_element_speed(mFields, elem);
    Omega_h::Vector<nodesPerElem> volHamiltonianCoeffs; // K_i in Barth-Sethian
    Plato::Scalar sumNegCoeffs = 0.;
    Plato::Scalar sumPosCoeffs = 0.;
    Plato::Scalar sumNegContrib = 0.;
    Plato::Scalar volHamiltonian = 0.;
    for (int n=0; n<nodesPerElem; ++n)
    {
      auto node = mElems2Verts[elem * nodesPerElem + n];
      const Plato::Scalar LSOld = mFields.mLevelSet(node, mFields.mCurrentState);
      volHamiltonianCoeffs[n] = elemVolume*elementSpeed * dot(normalDir, gradients[n]);
      volHamiltonian += volHamiltonianCoeffs[n] * LSOld;

      if (volHamiltonianCoeffs[n] < 0.)
      {
        sumNegCoeffs += volHamiltonianCoeffs[n];
        sumNegContrib += volHamiltonianCoeffs[n] * LSOld;
      }
      else
      {
        sumPosCoeffs += volHamiltonianCoeffs[n];
      }
    }

    Omega_h::Vector<nodesPerElem> alpha; // delta phi_i in Barth-Sethian
    Plato::Scalar sumPosAlpha = 0.;
    for (int n=0; n<nodesPerElem; ++n)
    {
      alpha[n] = 0.;
      auto node = mElems2Verts[elem * nodesPerElem + n];
      const Plato::Scalar LSOld = mFields.mLevelSet(node, mFields.mCurrentState);
      if (volHamiltonianCoeffs[n] > 0.)
      {
        alpha[n] = volHamiltonianCoeffs[n]/sumPosCoeffs*(sumNegContrib-sumNegCoeffs*LSOld)/volHamiltonian;
        if (alpha[n] > 0.) sumPosAlpha += alpha[n];
      }
    }

    for (int n=0; n<nodesPerElem; ++n)
    {
      auto node = mElems2Verts[elem * nodesPerElem + n];
      const Plato::Scalar LSOld = mFields.mLevelSet(node, mFields.mCurrentState);

      if (alpha[n] > 0.)
      {
        const Plato::Scalar wt = alpha[n]/sumPosAlpha;
        Kokkos::atomic_add(&mFields.mRHS(node), wt * volHamiltonian);
        Kokkos::atomic_add(&mFields.mRHSNorm(node), wt * elemVolume);
      }
    }
  }
};

template<int SpatialDim>
class AssembleNodalSpeedHamiltonian
{
  // Uses nodal speed to assemble nodal contributions for Hamiltonian.
  // Note that while the form supports any kind of nodal speed, this is currently specialized to
  // support Eikonal-type nodal speeds in get_nodal_speed().
private:
  const Omega_h::LOs mElems2Verts;
  const Plato::NodeCoordinate<SpatialDim> mNodeCoordinate;
  const Plato::ComputeGradient<SpatialDim> mComputeGradient;
  const ProblemFields<SpatialDim> mFields;
  const Plato::Scalar mEps;
  const bool mComputeTimeOfArrival;

public:
  AssembleNodalSpeedHamiltonian(Omega_h::Mesh & omega_h_mesh,
      ProblemFields<SpatialDim> & fields,
      const Plato::Scalar eps,
      const bool computeTimeOfArrival) :
        mElems2Verts(omega_h_mesh.ask_elem_verts()),
        mNodeCoordinate(&omega_h_mesh),
        mComputeGradient(mNodeCoordinate),
        mFields(fields),
        mEps(eps),
        mComputeTimeOfArrival(computeTimeOfArrival) {}


  DEVICE_TYPE inline
  Plato::Scalar get_nodal_speed(const ProblemFields<SpatialDim> & fields, int elem, int node, Plato::Scalar eps) const
  {
    // Note that this could be generalized to support some other type of nodal speed.
    // For now just implement Eikonal-type nodal speed (either redistancing or time-of-arrival).
    auto tOldLevelSet = fields.mLevelSet(node, fields.mCurrentState);
    auto tValue = (tOldLevelSet * tOldLevelSet) + (eps * eps);
    auto tDenominator = sqrt(tValue);
    auto tSign = tOldLevelSet / tDenominator;
    auto tSpeed = ( fields.mUseElementSpeed ? tSign*fields.mElementSpeed[elem] : tSign*fields.mNodalSpeed(node) );
    auto tOutput = mComputeTimeOfArrival ? tSpeed : tSign;
    return (tOutput);
  }

  DEVICE_TYPE inline
  void operator()(int elem) const
  {
    constexpr int nodesPerElem = SpatialDim + 1;
    Plato::Scalar elemVolume;
    Omega_h::Vector<SpatialDim> gradients[nodesPerElem];
    mComputeGradient(elem, gradients, elemVolume);

    Omega_h::Vector<SpatialDim> normalDir;
    for (int dim=0; dim<SpatialDim; ++dim) normalDir[dim] = 0.;
    for (int n=0; n<nodesPerElem; ++n)
    {
      const Plato::Scalar LSOld = mFields.mLevelSet(mElems2Verts[elem * nodesPerElem + n], mFields.mCurrentState);
      for (int dim=0; dim<SpatialDim; ++dim) normalDir[dim] += gradients[n][dim] * LSOld;
    }
    unitize(normalDir);

    for (int n=0; n<nodesPerElem; ++n)
    {
      auto node = mElems2Verts[elem * nodesPerElem + n];

      const Plato::Scalar nodalSpeed = get_nodal_speed(mFields, elem, node, mEps);
      const Plato::Scalar volHamiltonianCoeffs_n = elemVolume*nodalSpeed * dot(normalDir, gradients[n]);

      if (volHamiltonianCoeffs_n > 0.)
      {
        const Plato::Scalar LSOld = mFields.mLevelSet(node, mFields.mCurrentState);
        Plato::Scalar sumNegCoeffs = 0.;
        Plato::Scalar sumPosCoeffs = 0.;
        Plato::Scalar sumNegContrib = 0.;
        Plato::Scalar volHamiltonian = 0.;
        for (int j=0; j<nodesPerElem; ++j)
        {
          const Plato::Scalar LSOldj = mFields.mLevelSet(mElems2Verts[elem * nodesPerElem + j], mFields.mCurrentState);
          const Plato::Scalar volHamiltonianCoeffs_j = elemVolume*nodalSpeed * dot(normalDir, gradients[j]);

          volHamiltonian += volHamiltonianCoeffs_j * LSOldj;

          if (volHamiltonianCoeffs_j < 0.)
          {
            sumNegCoeffs += volHamiltonianCoeffs_j;
            sumNegContrib += volHamiltonianCoeffs_j * LSOldj;
          }
          else
          {
            sumPosCoeffs += volHamiltonianCoeffs_j;
          }
        }
        const Plato::Scalar wt = volHamiltonianCoeffs_n/sumPosCoeffs*(sumNegContrib-sumNegCoeffs*LSOld)/volHamiltonian;
        if (wt > 0.)
        {
          Kokkos::atomic_add(&mFields.mRHS(node), wt * volHamiltonian);
          Kokkos::atomic_add(&mFields.mRHSNorm(node), wt * elemVolume);
        }
      }
    }
  }
};

template<int SpatialDim>
void zero_RHS(Omega_h::Mesh & aOmega_h_Mesh, ProblemFields<SpatialDim> & aFields)
{
  auto RHS = aFields.mRHS;
  auto zeroRHS = LAMBDA_EXPRESSION(int n) { RHS(n) = 0.; };
  Kokkos::parallel_for(aOmega_h_Mesh.nverts(), zeroRHS);

  auto RHSNorm = aFields.mRHSNorm;
  auto zeroRHSNorm = LAMBDA_EXPRESSION(int n) { RHSNorm(n) = 0.; };
  Kokkos::parallel_for(aOmega_h_Mesh.nverts(), zeroRHSNorm);
}

template<int SpatialDim>
Plato::Scalar assemble_and_update_Eikonal(
    Omega_h::Mesh & aMeshOmegaH,
    ProblemFields<SpatialDim> & aFields,
    const Plato::Scalar aEps,
    const Plato::Scalar aDeltaTime,
    const bool aComputeArrivalTime)
{
  // This is a hybrid between the Barth-Sethian positive coefficient scheme and the Morgan-Waltz scheme for reinitialization.
  // Uses element based speed and nodal sign function to assemble nodal contributions for Hamiltonian (AssembleEikonalHamiltonian).
  // The assembled nodal Hamiltonian is then used with nodal source term to explicitly update signed distance
  // (or arrival time for non-unit speed).
  // Unlike the elemental algorithm developed by Barth-Sethian, this algorithm converges to the exact solution
  // for the "Distance Function Test" described in Morgan-Waltz.  Unlike the Morgan-Waltz algorithm, this
  // form converges much faster and is tolerant of meshes with obtuse angles because it uses the positive coefficient
  // form in Barth-Sethian.

  zero_RHS(aMeshOmegaH, aFields);

  AssembleNodalSpeedHamiltonian<SpatialDim> tAssemble(aMeshOmegaH, aFields, aEps, aComputeArrivalTime);
  Kokkos::parallel_for(aMeshOmegaH.nelems(), tAssemble);

  auto tLevelSet = aFields.mLevelSet;
  auto tCurrentState = aFields.mCurrentState;
  auto tNextState = (tCurrentState + 1) % 2;
  auto tUpdateLevelSetAndReduceResidual = LAMBDA_EXPRESSION(int aIndex, SumArray<Plato::Scalar, 2> & aResidAndCount)
  {
      auto tEpsTimesEps = aEps*aEps;
      auto tLevelSetTimesLevelSet = tLevelSet(aIndex,tCurrentState) * tLevelSet(aIndex,tCurrentState);
      auto tValue = tLevelSetTimesLevelSet + tEpsTimesEps;
      auto tSqrt = sqrt(tValue);
      auto tSign = tLevelSet(aIndex,tCurrentState) / tSqrt;

      tValue = aFields.mRHS(aIndex) / aFields.mRHSNorm(aIndex);
      auto tCondition = aFields.mRHSNorm(aIndex) > 0.;
      auto tHamiltonian = tCondition ? tValue : 0.;
      tValue = aDeltaTime * (tHamiltonian - tSign);
      tLevelSet(aIndex,tNextState) = tLevelSet(aIndex,tCurrentState) - tValue;

      tCondition = abs(tLevelSet(aIndex,tNextState)) < aEps;
      if (aComputeArrivalTime || tCondition)
      {
          auto tMyValue = (tHamiltonian - tSign)*(tHamiltonian - tSign);
          aResidAndCount[0] += tMyValue;
          aResidAndCount[1] += 1.0;
      }
  };

  SumArray<Plato::Scalar, 2> tResidAndCount;
  Kokkos::parallel_reduce(aMeshOmegaH.nverts(), tUpdateLevelSetAndReduceResidual, tResidAndCount);

  aFields.mCurrentState = tNextState;

  auto tValue = tResidAndCount[0] / tResidAndCount[1];
  return std::sqrt(tValue);
}

template<int SpatialDim>
void evolve_level_set(
    Omega_h::Mesh & aMeshOmegaH,
    ProblemFields<SpatialDim> & aFields,
    const Plato::Scalar aEps,
    const Plato::Scalar aDeltaTime)
{
  // This is a uses the Barth-Sethian positive coefficient scheme for advection.

  zero_RHS(aMeshOmegaH, aFields);

  AssembleElementSpeedHamiltonian<SpatialDim> tAssemble(aMeshOmegaH, aFields, aEps);
  Kokkos::parallel_for(aMeshOmegaH.nelems(), tAssemble);

  const int tCurrentState = aFields.mCurrentState;
  const int tNextState = (tCurrentState+1)%2;
  auto tLevelSet = aFields.mLevelSet;
  auto updateLevelSet = LAMBDA_EXPRESSION(int aN)
  {
      Plato::Scalar tValueTwo = 0.;
      Plato::Scalar tValueOne = aFields.mRHS(aN)/aFields.mRHSNorm(aN);
      auto tHamiltonian = (aFields.mRHSNorm(aN) > 0.) ? tValueOne : tValueTwo;
      tLevelSet(aN,tNextState) = tLevelSet(aN,tCurrentState) - (aDeltaTime * tHamiltonian);
  };
  Kokkos::parallel_for(aMeshOmegaH.nverts(), updateLevelSet);

  aFields.mCurrentState = tNextState;
}

template<int SpatialDim>
void write_mesh(Omega_h::vtk::Writer & aWriter,
                Omega_h::Mesh & aOmegaH_Mesh,
                ProblemFields<SpatialDim> & aFields,
                const Plato::Scalar aTime)
{
    const Plato::OrdinalType tNodeCount = aOmegaH_Mesh.nverts();
    Kokkos::View<Omega_h::Real*> tView("into", tNodeCount);
    auto tSubView = Kokkos::subview(aFields.mLevelSet, Kokkos::ALL(), aFields.mCurrentState);
    Kokkos::deep_copy(tView, tSubView);
    aOmegaH_Mesh.add_tag(Omega_h::VERT, "LevelSet", 1, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(tView)));
    auto tTags = Omega_h::vtk::get_all_vtk_tags(&aOmegaH_Mesh, SpatialDim);
    aWriter.write(static_cast<Omega_h::Real>(aTime), tTags);
}

template<int SpatialDim>
void compute_arrival_time (
    Omega_h::Mesh & omega_h_mesh,
    ProblemFields<SpatialDim> & fields,
    const Plato::Scalar eps,
    const Plato::Scalar dtau,
    const Plato::Scalar convergedTol = 0.01)
{
  if (!domain_contains_interface(omega_h_mesh, fields))
  {
    std::cout << "Error, input level set field does not contain zero level set for initializing arrival time calculation." << std::endl;
    return;
  }
  Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer("MyMesh", &omega_h_mesh, SpatialDim);

  bool converged = false;
  const int maxIters = 5000;
  const int printFreq = 100;
  for (int iter = 0; iter<maxIters; ++iter)
  {
    const Plato::Scalar averageNodalResidual =  assemble_and_update_Eikonal(omega_h_mesh, fields, eps, dtau, true);

    if ((iter+1) % printFreq == 0)
    {
      std::cout << "After " << iter+1 << " iterations in compute_arrival_time, the relative error is " << averageNodalResidual << std::endl;
      write_mesh(tWriter, omega_h_mesh, fields, iter*dtau);
    }

    if (averageNodalResidual < convergedTol)
    {
      converged = true;
      std::cout << "compute_arrival_time converged after " << iter+1 << " iterations  with relative error of " << averageNodalResidual << std::endl;
      break;
    }
  }
  if (!converged)
  {
    std::cout << "compute_arrival_time failed to converge after " << maxIters << " iterations." << std::endl;
  }
}

template<int SpatialDim>
void reinitialize_level_set (
    Omega_h::Mesh & aMeshOmegaH,
    ProblemFields<SpatialDim> & aFields,
    const Plato::Scalar aTime,
    const Plato::Scalar aEps,
    const Plato::Scalar aDeltaTau,
    const Plato::OrdinalType aMaxIters = 10,
    const Plato::Scalar aConvergedTol = 0.01)
{
    if(!domain_contains_interface(aMeshOmegaH, aFields))
    {
        return;
    }

    bool converged = false;
    const int printFreq = 100;
    for(int iter = 0; iter < aMaxIters; ++iter)
    {
        const Plato::Scalar averageNodalResidual = assemble_and_update_Eikonal(aMeshOmegaH, aFields, aEps, aDeltaTau, false);

        if((iter + 1) % printFreq == 0)
        {
            std::cout << "After " << iter + 1 << " iterations in reinitialize_level_set, the relative error is "
            << averageNodalResidual << std::endl;
        }

        if(averageNodalResidual < aConvergedTol)
        {
            converged = true;
            std::cout << "At time " << aTime << ", reinitialize_level_set converged after " << iter + 1
            << " iterations  with relative error of " << averageNodalResidual << std::endl;
            break;
        }
    }
    if(!converged)
    {
        std::cout << "At time " << aTime << ", reinitialize_level_set failed to converge after " << aMaxIters
        << " iterations." << std::endl;
    }
}
#endif
