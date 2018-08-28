#ifndef TENSOR_P_NORM_HPP
#define TENSOR_P_NORM_HPP

#include "plato/PlatoStaticsTypes.hpp"

/******************************************************************************/
/*! Tensor p-norm functor.
  
    Given a voigt tensors, compute the p-norm.
    Assumes single point integration.
*/
/******************************************************************************/
template<Plato::OrdinalType VoigtLength>
class TensorPNormFunctor
{
  public:

    TensorPNormFunctor(Teuchos::ParameterList& params){}

    template<typename ResultScalarType, 
             typename TensorScalarType, 
             typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarVectorT<ResultScalarType> pnorm,
                Plato::ScalarMultiVectorT<TensorScalarType> voigtTensor,
                Plato::OrdinalType p,
                Plato::ScalarVectorT<VolumeScalarType> cellVolume ) const {

      // compute scalar product
      //
      pnorm(cellOrdinal) = 0.0;
      for( Plato::OrdinalType iVoigt=0; iVoigt<VoigtLength; iVoigt++){
        pnorm(cellOrdinal) += voigtTensor(cellOrdinal,iVoigt)*voigtTensor(cellOrdinal,iVoigt);
      }
      pnorm(cellOrdinal) = pow(pnorm(cellOrdinal),p/2.0);
      pnorm(cellOrdinal) *= cellVolume(cellOrdinal);
    }
};

/******************************************************************************/
/*! Weighted tensor p-norm functor.
  
    Given a voigt tensors, compute the weighted p-norm.
    Assumes single point integration.
*/
/******************************************************************************/
template<Plato::OrdinalType VoigtLength>
class WeightedNormFunctor
{

    Plato::Scalar m_scalarProductWeight;
    Plato::Scalar m_referenceWeight;
    
  public:

    WeightedNormFunctor(Teuchos::ParameterList& params) :
      m_scalarProductWeight(1.0),
      m_referenceWeight(1.0)
    {
      auto p = params.sublist("Normalize").sublist("Scalar");
      if( p.isType<double>("Reference Weight") ){
        m_referenceWeight = p.get<double>("Reference Weight");
      }
      if( p.isType<double>("Scalar Product Weight") ){
        m_scalarProductWeight = p.get<double>("Scalar Product Weight");
      }
    }

    template<typename ResultScalarType, 
             typename TensorScalarType, 
             typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarVectorT<ResultScalarType> pnorm,
                Plato::ScalarMultiVectorT<TensorScalarType> voigtTensor,
                Plato::OrdinalType p,
                Plato::ScalarVectorT<VolumeScalarType> cellVolume ) const {

      // compute scalar product
      //
      pnorm(cellOrdinal) = 0.0;
      for( Plato::OrdinalType iVoigt=0; iVoigt<VoigtLength; iVoigt++){
        pnorm(cellOrdinal) += voigtTensor(cellOrdinal,iVoigt)*voigtTensor(cellOrdinal,iVoigt);
      }
      pnorm(cellOrdinal) = sqrt(m_scalarProductWeight*pnorm(cellOrdinal));
      pnorm(cellOrdinal) /= m_referenceWeight;

      pnorm(cellOrdinal) = pow(pnorm(cellOrdinal),p);
      pnorm(cellOrdinal) *= cellVolume(cellOrdinal);
    }
};

/******************************************************************************/
/*! Barlat tensor p-norm functor.
  
    Given a voigt tensors, compute the Barlat p-norm.
    Assumes single point integration.
*/
/******************************************************************************/
template<Plato::OrdinalType VoigtLength>
class BarlatNormFunctor
{
    Plato::OrdinalType m_barlatExponent;
    Plato::Scalar m_scalarProductWeight;
    Plato::Scalar m_referenceWeight;

    static constexpr Plato::OrdinalType barlatLength = 6;
    Plato::Scalar m_cp[barlatLength][barlatLength];
    Plato::Scalar m_cpp[barlatLength][barlatLength];
    
  public:

    BarlatNormFunctor(Teuchos::ParameterList& params) :
      m_barlatExponent(6),
      m_referenceWeight(1.0)
    {

      auto p = params.sublist("Normalize").sublist("Barlat");
      if( p.isType<Plato::OrdinalType>("Barlat Exponent") ){
        m_barlatExponent = p.get<double>("Barlat Exponent");
      } 
      if( p.isType<double>("Reference Weight") ){
        m_referenceWeight = p.get<double>("Reference Weight");
      }
      
      Plato::Scalar cp[barlatLength][barlatLength];
      Plato::Scalar cpp[barlatLength][barlatLength];

      for(Plato::OrdinalType i=0; i<barlatLength; i++){
        std::stringstream ss; ss << "T1" << i;
        auto vals = p.get<Teuchos::Array<double>>(ss.str());
        for(Plato::OrdinalType j=0; j<barlatLength; j++) cp[i][j] = vals[j];
      }
         
      for(Plato::OrdinalType i=0; i<barlatLength; i++){
        std::stringstream ss; ss << "T2" << i;
        auto vals = p.get<Teuchos::Array<double>>(ss.str());
        for(Plato::OrdinalType j=0; j<barlatLength; j++) cpp[i][j] = vals[j];
      }

      Plato::Scalar mapToDevStress[barlatLength][barlatLength] =
        {  2.0/3.0, -1.0/3.0, -1.0/3.0, 0.0, 0.0, 0.0 ,
          -1.0/3.0,  2.0/3.0, -1.0/3.0, 0.0, 0.0, 0.0 ,
          -1.0/3.0, -1.0/3.0,  2.0/3.0, 0.0, 0.0, 0.0 ,
          0.0,       0.0,      0.0,     1.0, 0.0, 0.0 ,
          0.0,       0.0,      0.0,     0.0, 1.0, 0.0 ,
          0.0,       0.0,      0.0,     0.0, 0.0, 1.0 };

      for(Plato::OrdinalType i=0; i<barlatLength; i++){
        for(Plato::OrdinalType j=0; j<barlatLength; j++){
          m_cp[i][j]=0.0;
          m_cpp[i][j]=0.0;
          for(Plato::OrdinalType k=0; k<barlatLength; k++){
            m_cp[i][j]  += cp[i][k]*mapToDevStress[k][j];
            m_cpp[i][j] += cpp[i][k]*mapToDevStress[k][j];
          }
        }
      }
      
         
    }

    template<typename ResultScalarType, 
             typename TensorScalarType, 
             typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarVectorT<ResultScalarType> pnorm,
                Plato::ScalarMultiVectorT<TensorScalarType> voigtTensor,
                Plato::OrdinalType p,
                Plato::ScalarVectorT<VolumeScalarType> cellVolume ) const
    {
      TensorScalarType s[barlatLength]={0.0};
      if(VoigtLength == 6 ){
        for(Plato::OrdinalType i=0; i<VoigtLength; i++) s[i] = voigtTensor(cellOrdinal,i);
      } else
      if(VoigtLength == 3 ){
        s[0] = voigtTensor(cellOrdinal,0);
        s[1] = voigtTensor(cellOrdinal,1);
        s[5] = voigtTensor(cellOrdinal,2);
      } else
      if(VoigtLength == 1 ){
        s[0] = voigtTensor(cellOrdinal,0);
      }

      TensorScalarType sp[barlatLength], spp[barlatLength];
      for(Plato::OrdinalType i=0; i<barlatLength; i++){
        sp[i] = 0.0; spp[i] = 0.0;
        for(Plato::OrdinalType j=0; j<barlatLength; j++){
          sp[i]  += m_cp[i][j]*s[j];
          spp[i] += m_cpp[i][j]*s[j];
        }
      }

      //Get invariates of the deviators. appendix A of Barlat, 2004.
      TensorScalarType Hp1 = (sp[0]+sp[1]+sp[2]) / 3.0;
      TensorScalarType Hp2 = (sp[3]*sp[3]+sp[4]*sp[4]+sp[5]*sp[5]-sp[1]*sp[2]-sp[2]*sp[0]-sp[0]*sp[1]) / 3.0;
      TensorScalarType Hp3 = (2.0*sp[3]*sp[4]*sp[5]+sp[0]*sp[1]*sp[2]-sp[0]*sp[3]*sp[3]-sp[1]*sp[4]*sp[4]-sp[2]*sp[5]*sp[5]) / 2.0;
      TensorScalarType Hpp1 = (spp[0]+spp[1]+spp[2]) / 3.0;
      TensorScalarType Hpp2 = (spp[3]*spp[3]+spp[4]*spp[4]+spp[5]*spp[5]-spp[1]*spp[2]-spp[2]*spp[0]-spp[0]*spp[1]) / 3.0;
      TensorScalarType Hpp3 = (2.0*spp[3]*spp[4]*spp[5]+spp[0]*spp[1]*spp[2]-spp[0]*spp[3]*spp[3]-spp[1]*spp[4]*spp[4]-spp[2]*spp[5]*spp[5]) / 2.0;
    
      //Get interim values for p, q, and theta. These are defined analytically in the Barlat text.
      TensorScalarType Pp = 0.0;
      if (Hp1*Hp1+Hp2>0.0)
        Pp = Hp1*Hp1+Hp2;
      TensorScalarType Qp = (2.0*Hp1*Hp1*Hp1 + 3.0*Hp1*Hp2 + 2.0*Hp3) / 2.0;
      TensorScalarType Thetap = acos(Qp/pow(Pp,3.0/2.0));
    
      TensorScalarType Ppp = 0.0;
      if (Hpp1*Hpp1+Hpp2>0.0)
        Ppp = Hpp1*Hpp1+Hpp2;
      TensorScalarType Qpp = (2.0*Hpp1*Hpp1*Hpp1 + 3.0*Hpp1*Hpp2 + 2.0*Hpp3) / 2.0;
      TensorScalarType Thetapp = acos( Qpp / pow(Ppp,3.0/2.0) );
    
      //Apply the analytic solutions for the SVD stress deviators
      TensorScalarType Sp1 = (Hp1 + 2.0*sqrt(Hp1*Hp1+Hp2)*cos(Thetap/3.0));
      TensorScalarType Sp2 = (Hp1 + 2.0*sqrt(Hp1*Hp1+Hp2)*cos((Thetap+4.0*acos(-1.0))/3.0));
      TensorScalarType Sp3 = (Hp1 + 2.0*sqrt(Hp1*Hp1+Hp2)*cos((Thetap+2.0*acos(-1.0))/3.0));
    
      TensorScalarType Spp1 = (Hpp1 + 2.0*sqrt(Hpp1*Hpp1+Hpp2)*cos(Thetapp/3.0));
      TensorScalarType Spp2 = (Hpp1 + 2.0*sqrt(Hpp1*Hpp1+Hpp2)*cos((Thetapp+4.0*acos(-1.0))/3.0));
      TensorScalarType Spp3 = (Hpp1 + 2.0*sqrt(Hpp1*Hpp1+Hpp2)*cos((Thetapp+2.0*acos(-1.0))/3.0));
    
      //Apply the Barlat yield function to get effective response.
      pnorm(cellOrdinal) = pow( (pow(Sp1-Spp1,m_barlatExponent)+
                                 pow(Sp1-Spp2,m_barlatExponent)+
                                 pow(Sp1-Spp3,m_barlatExponent)+
                                 pow(Sp2-Spp1,m_barlatExponent)+
                                 pow(Sp2-Spp2,m_barlatExponent)+
                                 pow(Sp2-Spp3,m_barlatExponent)+
                                 pow(Sp3-Spp1,m_barlatExponent)+
                                 pow(Sp3-Spp2,m_barlatExponent)+
                                 pow(Sp3-Spp3,m_barlatExponent) )/4.0 , 1.0/m_barlatExponent);


      // normalize
      pnorm(cellOrdinal) /= m_referenceWeight;

      // pnorm
      pnorm(cellOrdinal) = pow(pnorm(cellOrdinal),p);
      pnorm(cellOrdinal) *= cellVolume(cellOrdinal);
  }
};



/******************************************************************************/
/*! Abstract base class for computing a tensor norm.
*/
/******************************************************************************/
template<Plato::OrdinalType VoigtLength, typename EvalT>
class TensorNormBase {

  protected:
    Plato::Scalar m_exponent;

  public:
    TensorNormBase(Teuchos::ParameterList& params){
      m_exponent = params.get<double>("Exponent");
    }

    virtual void
    evaluate(Plato::ScalarVectorT<typename EvalT::ResultScalarType> result,
             Plato::ScalarMultiVectorT<typename EvalT::ResultScalarType> tensor,
             Plato::ScalarMultiVectorT<typename EvalT::ControlScalarType> control,
             Plato::ScalarVectorT<typename EvalT::ConfigScalarType> cellVolume) const = 0;

    virtual void
    postEvaluate( Plato::ScalarVector resultVector,
                  Plato::Scalar       resultScalar)
    {
      auto scale = pow(resultScalar,(1.0-m_exponent)/m_exponent)/m_exponent;
      auto numEntries = resultVector.size();
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,numEntries), LAMBDA_EXPRESSION(Plato::OrdinalType entryOrdinal)
      {
        resultVector(entryOrdinal) *= scale;
      },"scale vector");
    }

    virtual void
    postEvaluate( Plato::Scalar& resultValue )
    {
      resultValue = pow(resultValue, 1.0/m_exponent);
    }

};

template<Plato::OrdinalType VoigtLength, typename EvalT>
class TensorPNorm : public TensorNormBase<VoigtLength, EvalT> {

  TensorPNormFunctor<VoigtLength> m_tensorPNorm;

  public:

    TensorPNorm(Teuchos::ParameterList& params) : 
      TensorNormBase<VoigtLength, EvalT>(params),
      m_tensorPNorm(params) {}

    void
    evaluate(Plato::ScalarVectorT<typename EvalT::ResultScalarType> result,
             Plato::ScalarMultiVectorT<typename EvalT::ResultScalarType> tensor,
             Plato::ScalarMultiVectorT<typename EvalT::ControlScalarType> control,
             Plato::ScalarVectorT<typename EvalT::ConfigScalarType> cellVolume) const
    {
      Plato::OrdinalType numCells = result.extent(0);
      auto exponent = TensorNormBase<VoigtLength, EvalT>::m_exponent;
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
        // compute tensor p-norm of tensor
        //
        m_tensorPNorm(cellOrdinal, result, tensor, exponent, cellVolume);

      },"Compute PNorm");
    }
};

template<Plato::OrdinalType VoigtLength, typename EvalT>
class BarlatNorm : public TensorNormBase<VoigtLength, EvalT> {

    BarlatNormFunctor<VoigtLength> m_barlatNorm;

  public:

    BarlatNorm(Teuchos::ParameterList& params) : 
      TensorNormBase<VoigtLength, EvalT>(params),
      m_barlatNorm(params){}

    void
    evaluate(Plato::ScalarVectorT<typename EvalT::ResultScalarType> result,
             Plato::ScalarMultiVectorT<typename EvalT::ResultScalarType> tensor,
             Plato::ScalarMultiVectorT<typename EvalT::ControlScalarType> control,
             Plato::ScalarVectorT<typename EvalT::ConfigScalarType> cellVolume) const
    {
      Plato::OrdinalType numCells = result.extent(0);
      auto exponent = TensorNormBase<VoigtLength, EvalT>::m_exponent;
      auto barlatNorm = m_barlatNorm;
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
        // compute tensor p-norm of tensor
        //
        barlatNorm(cellOrdinal, result, tensor, exponent, cellVolume);

      },"Compute Barlat Norm");
    }
};

template<Plato::OrdinalType VoigtLength, typename EvalT>
class WeightedNorm : public TensorNormBase<VoigtLength, EvalT> {

    WeightedNormFunctor<VoigtLength> m_weightedNorm;

  public:

    WeightedNorm(Teuchos::ParameterList& params) : 
      TensorNormBase<VoigtLength, EvalT>(params),
      m_weightedNorm(params){}

    void
    evaluate(Plato::ScalarVectorT<typename EvalT::ResultScalarType> result,
             Plato::ScalarMultiVectorT<typename EvalT::ResultScalarType> tensor,
             Plato::ScalarMultiVectorT<typename EvalT::ControlScalarType> control,
             Plato::ScalarVectorT<typename EvalT::ConfigScalarType> cellVolume) const
    {
      Plato::OrdinalType numCells = result.extent(0);
      auto exponent = TensorNormBase<VoigtLength, EvalT>::m_exponent;
      auto weightedNorm = m_weightedNorm;
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
        // compute tensor p-norm of tensor
        //
        weightedNorm(cellOrdinal, result, tensor, exponent, cellVolume);

      },"Compute Weighted Norm");
    }
};


template<Plato::OrdinalType VoigtLength, typename EvalT>
struct TensorNormFactory {
  
  Teuchos::RCP<TensorNormBase<VoigtLength, EvalT>>
  create(Teuchos::ParameterList params){

    Teuchos::RCP<TensorNormBase<VoigtLength, EvalT>> retval = Teuchos::null;
 
    if( params.isSublist("Normalize") ){
      auto normList = params.sublist("Normalize");
      auto normType = normList.get<std::string>("Type");
      if( normType == "Barlat" ){
        retval = Teuchos::rcp(new BarlatNorm<VoigtLength, EvalT>(params));
      } else
      if( normType == "Scalar" ){
        retval = Teuchos::rcp(new WeightedNorm<VoigtLength, EvalT>(params));
      } else {
        retval = Teuchos::rcp(new TensorPNorm<VoigtLength, EvalT>(params));
      }
    } else {
      retval = Teuchos::rcp(new TensorPNorm<VoigtLength, EvalT>(params));
    }

    return retval;
  }
};

#endif
