#pragma once

#include "plato/PlatoStaticsTypes.hpp"
#include "plato/PlatoTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Compute Eigenvalues
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class Eigenvalues
{
private:
    Plato::Scalar mTolerance;
    Plato::OrdinalType mMaxIterations;

public:
	/******************************************************************************//**
     * @brief Constructor
    **********************************************************************************/
    Eigenvalues(Plato::Scalar aTolerance = 1.0e-12, Plato::OrdinalType aMaxIterations = 20) :
    mTolerance(aTolerance),
    mMaxIterations(aMaxIterations)
    {
    }

    /******************************************************************************//**
     * @brief Destructor
    **********************************************************************************/
    ~Eigenvalues(){}

    /******************************************************************************//**
     * @brief Set tolerance
     * @param [in] aTolerance new tolerance for jacobi eigenvalue solver
    **********************************************************************************/
    void setTolerance(const Plato::Scalar aTolerance)
    {
    	mTolerance = aTolerance;
    }

    /******************************************************************************//**
     * @brief Set maximum number of jacobi iterations
     * @param [in] aMaxIterations maximum number of iterations for jacobi eigenvalue solver
    **********************************************************************************/
    void setMaxIterations(const Plato::Scalar aMaxIterations)
    {
    	mMaxIterations = aMaxIterations;
    }

    /******************************************************************************//**
     * @brief Compute Eigenvalues
     * @param [in] aCellOrdinal cell/element index
     * @param [in] aVoigtTensor cell/element voigt tensor
     * @param [in] aIsStrainType engineering factor - divide shear terms by 2
     * @param [out] aEigenvalues cell/element tensor eigenvalues
    **********************************************************************************/
    template<typename InputType, typename ResultType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVectorT<InputType> & aVoigtTensor,
               const Plato::ScalarMultiVectorT<ResultType> & aEigenvalues,
               const bool & aIsStrainType) const;
};
// class Eigenvalues

/******************************************************************************//**
 * @brief Eigenvalues for 3D problems
 *
 * @param [in] aCellOrdinal cell/element local ordinal
 * @param [in] aVoigtTensor cell/element voigt tensor
 * @param [in] aIsStrainType engineering factor - divide shear terms by 2
 * @param [out] aEigenvalues cell/element tensor eigenvalues
**********************************************************************************/
template<>
template<typename InputType, typename ResultType>
DEVICE_TYPE inline void
Eigenvalues<3>::operator()(const Plato::OrdinalType & aCellOrdinal,
                           const Plato::ScalarMultiVectorT<InputType> & aVoigtTensor,
                           const Plato::ScalarMultiVectorT<ResultType> & aEigenvalues,
                           const bool & aIsStrainType) const
{
    ResultType tTensor[3][3];

    // Fill diagonal elements
    tTensor[0][0] = aVoigtTensor(aCellOrdinal, 0);
    tTensor[1][1] = aVoigtTensor(aCellOrdinal, 1);
    tTensor[2][2] = aVoigtTensor(aCellOrdinal, 2);

    // Fill used off diagonal elements
    if (aIsStrainType)
    {
        tTensor[0][1] = aVoigtTensor(aCellOrdinal, 5) / static_cast<Plato::Scalar>(2.0);
        tTensor[0][2] = aVoigtTensor(aCellOrdinal, 4) / static_cast<Plato::Scalar>(2.0);
        tTensor[1][2] = aVoigtTensor(aCellOrdinal, 3) / static_cast<Plato::Scalar>(2.0);
    }
    else
    {
        tTensor[0][1] = aVoigtTensor(aCellOrdinal, 5);
        tTensor[0][2] = aVoigtTensor(aCellOrdinal, 4);
        tTensor[1][2] = aVoigtTensor(aCellOrdinal, 3);
    }
    // Symmetrize
    tTensor[1][0] = tTensor[0][1]; 
    tTensor[2][0] = tTensor[0][2]; 
    tTensor[2][1] = tTensor[1][2];

    ResultType tOffDiagNorm = sqrt(tTensor[0][1]*tTensor[0][1] + tTensor[0][2]*tTensor[0][2] + 
                                   tTensor[1][2]*tTensor[1][2]);

    ResultType tRelativeTolerance = mTolerance * sqrt(tTensor[0][0]*tTensor[0][0] + tTensor[1][1]*tTensor[1][1] 
                                                    + tTensor[2][2]*tTensor[2][2]);

    if (tRelativeTolerance < mTolerance)
        tRelativeTolerance = mTolerance;

    if (tOffDiagNorm < tRelativeTolerance)
    {
        aEigenvalues(aCellOrdinal, 0) = aVoigtTensor(aCellOrdinal, 0);
        aEigenvalues(aCellOrdinal, 1) = aVoigtTensor(aCellOrdinal, 1);
        aEigenvalues(aCellOrdinal, 2) = aVoigtTensor(aCellOrdinal, 2);
    }
    else // Start Jacobi Iteration
    {
        ResultType tGivensRotation[3][3];
        ResultType tTensorRotated[3][3];
        ResultType tSine, tCosine, tTangent, tTau;
        Plato::OrdinalType i, j, k, l, p, q;
        Plato::OrdinalType tIteration = static_cast<Plato::OrdinalType>(0);
        while ((tOffDiagNorm > tRelativeTolerance) && (tIteration < mMaxIterations))
        {
            // ########## Compute location of max off-diagonal entry ##########
            if (abs(tTensor[0][1]) >= abs(tTensor[0][2]))
            {
                if (abs(tTensor[0][1]) >= abs(tTensor[1][2]))
                {   p = static_cast<Plato::OrdinalType>(0); q = static_cast<Plato::OrdinalType>(1);  }
                else
                {   p = static_cast<Plato::OrdinalType>(1); q = static_cast<Plato::OrdinalType>(2);  }
            }
            else
            {
                if (abs(tTensor[0][2]) >= abs(tTensor[1][2]))
                {   p = static_cast<Plato::OrdinalType>(0); q = static_cast<Plato::OrdinalType>(2);  }
                else
                {   p = static_cast<Plato::OrdinalType>(1); q = static_cast<Plato::OrdinalType>(2);  }
            }

            // ########## Compute rotation sine and cosine ##########
            if (abs(tTensor[p][q]) > 1.0e-15)
            {
                tTau = (tTensor[q][q] - tTensor[p][p]) / (2.0 * tTensor[p][q]);
                if (tTau >= static_cast<Plato::Scalar>(0.0))
                {
                    tTangent =
                      static_cast<Plato::Scalar>( 1.0) / (tTau + sqrt(static_cast<Plato::Scalar>(1.0) + tTau*tTau));
                }
                else
                {
                    tTangent =
                        static_cast<Plato::Scalar>(-1.0) / (static_cast<Plato::Scalar>(-1.0) * tTau + 
                                                            sqrt(static_cast<Plato::Scalar>(1.0) + tTau*tTau));
                }

                tCosine = static_cast<Plato::Scalar>(1.0) / sqrt(static_cast<Plato::Scalar>(1.0) + tTangent*tTangent);
                tSine   = tTangent * tCosine;
            }
            else
            {
                tCosine = static_cast<Plato::Scalar>(1.0);
                tSine   = static_cast<Plato::Scalar>(0.0);
            }

            // ########## Apply similarity transform with Givens rotation ##########
            tGivensRotation[0][0] = 1.0; tGivensRotation[0][1] = 0.0; tGivensRotation[0][2] = 0.0;
            tGivensRotation[1][0] = 0.0; tGivensRotation[1][1] = 1.0; tGivensRotation[1][2] = 0.0;
            tGivensRotation[2][0] = 0.0; tGivensRotation[2][1] = 0.0; tGivensRotation[2][2] = 1.0;

            tGivensRotation[p][p] = tCosine;  tGivensRotation[p][q] =   tSine;
            tGivensRotation[q][p] =  -tSine;  tGivensRotation[q][q] = tCosine;

            for (i = 0; i < 3; ++i)
                for (l = i; l < 3; ++l) // Note that symmetry is being employed for speed
                {
                    tTensorRotated[i][l] = 0.0;
                    for (j = 0; j < 3; ++j)
                        for (k = 0; k < 3; ++k)
                            tTensorRotated[i][l] += tGivensRotation[j][i] * tTensor[j][k] * tGivensRotation[k][l];

                    tTensorRotated[l][i] = tTensorRotated[i][l];
                }

            for (i = 0; i < 3; ++i)
                for (j = 0; j < 3; ++j)
                    tTensor[i][j] = tTensorRotated[i][j];

            // ########## Recompute off-diagonal norm for convergence test ##########
            tOffDiagNorm = sqrt(tTensor[0][1]*tTensor[0][1] + tTensor[0][2]*tTensor[0][2] + 
                                tTensor[1][2]*tTensor[1][2]);

            ++tIteration;
        }

        aEigenvalues(aCellOrdinal, 0) = tTensor[0][0];
        aEigenvalues(aCellOrdinal, 1) = tTensor[1][1];
        aEigenvalues(aCellOrdinal, 2) = tTensor[2][2];
    }
}

/******************************************************************************//**
 * @brief Eigenvalues for 2D problems
 *
 * @param [in] aCellOrdinal cell/element local ordinal
 * @param [in] aVoigtTensor cell/element voigt tensor
 * @param [in] aIsStrainType engineering factor - divide shear terms by 2
 * @param [out] aEigenvalues cell/element tensor eigenvalues
**********************************************************************************/
template<>
template<typename InputType, typename ResultType>
DEVICE_TYPE inline void
Eigenvalues<2>::operator()(const Plato::OrdinalType & aCellOrdinal,
                           const Plato::ScalarMultiVectorT<InputType> & aVoigtTensor,
                           const Plato::ScalarMultiVectorT<ResultType> & aEigenvalues,
                           const bool & aIsStrainType) const
{
    ResultType tTensor12;
    if (aIsStrainType)
    {
        tTensor12 = (static_cast<Plato::Scalar>(0.5) * aVoigtTensor(aCellOrdinal, 2));
    }
    else
    {
        tTensor12 = aVoigtTensor(aCellOrdinal, 2);
    }
    
    if (abs(tTensor12) < mTolerance) // Tensor is diagonal
    {
        aEigenvalues(aCellOrdinal, 0) = aVoigtTensor(aCellOrdinal, 0);
        aEigenvalues(aCellOrdinal, 1) = aVoigtTensor(aCellOrdinal, 1);
    }
    else // 1 iteration of Jacobi is required
    {
        ResultType tTensor11 = aVoigtTensor(aCellOrdinal, 0);
        ResultType tTensor22 = aVoigtTensor(aCellOrdinal, 1);
        ResultType tTau = (tTensor22 - tTensor11) / (static_cast<Plato::Scalar>(2.0) * tTensor12);

        ResultType tTangent;
        if (tTau >= static_cast<Plato::Scalar>(0.0))
        {
            tTangent = static_cast<Plato::Scalar>( 1.0) / (tTau + sqrt(static_cast<Plato::Scalar>(1.0) + tTau*tTau));
        }
        else
        {
            tTangent = static_cast<Plato::Scalar>(-1.0) / (static_cast<Plato::Scalar>(-1.0) * tTau + 
                                                       sqrt(static_cast<Plato::Scalar>(1.0) + tTau*tTau));
        }

        ResultType tCosine = static_cast<Plato::Scalar>(1.0) / sqrt(static_cast<Plato::Scalar>(1.0) + tTangent*tTangent);
        ResultType tSine   = tTangent * tCosine;
        aEigenvalues(aCellOrdinal, 0) = tCosine * (tTensor11*tCosine - tTensor12*tSine) -
                                        tSine   * (tTensor12*tCosine - tTensor22*tSine);
        aEigenvalues(aCellOrdinal, 1) = tCosine * (tTensor22*tCosine + tTensor12*tSine) +
                                        tSine   * (tTensor12*tCosine + tTensor11*tSine);
    }
}

/******************************************************************************//**
 * @brief Eigenvalues for 1D problems
 *
 * @param [in] aCellOrdinal cell/element local ordinal
 * @param [in] aVoigtTensor cell/element voigt tensor
 * @param [in] aIsStrainType engineering factor - divide shear terms by 2
 * @param [out] aEigenvalues cell/element tensor eigenvalues
**********************************************************************************/
template<>
template<typename InputType, typename ResultType>
DEVICE_TYPE inline void
Eigenvalues<1>::operator()(const Plato::OrdinalType & aCellOrdinal,
                           const Plato::ScalarMultiVectorT<InputType> & aVoigtTensor,
                           const Plato::ScalarMultiVectorT<ResultType> & aEigenvalues,
                           const bool & aIsStrainType) const
{
    ResultType tOutput = aVoigtTensor(aCellOrdinal, 0);
    aEigenvalues(aCellOrdinal, 0) = tOutput;
}

}
// namespace Plato