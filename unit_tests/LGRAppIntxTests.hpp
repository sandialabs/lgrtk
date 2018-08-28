#include "plato/LGR_App.hpp"
#include <cstdlib>

class FauxSharedData {
  public:
    FauxSharedData(Plato::data::layout_t layout, int size, double initVal=0.0 ) : 
      m_data(size,initVal), m_layout(layout){}

    void setData(const std::vector<double> & aData)
    {
      m_data = aData;
    }
    void getData(std::vector<double> & aData) const
    {
      aData = m_data;
    }
    int size() const
    {
      return m_data.size();
    }

    std::string myContext() const {return m_context;}
    void setContext(std::string context) {m_context = context;}

    Plato::data::layout_t myLayout() const 
    {
      return m_layout;
    }

    double operator[](int index){ return m_data[index]; }

  protected:
    std::vector<double> m_data;
    Plato::data::layout_t m_layout;
    std::string m_context;
};
class FauxSharedField : public FauxSharedData
{
  public:
    FauxSharedField(int size, double initVal=0.0) : 
       FauxSharedData(Plato::data::layout_t::SCALAR_FIELD, size, initVal){}
};

class FauxSharedValue : public FauxSharedData
{
  public:
    FauxSharedValue(int size, double initVal=0.0) : 
       FauxSharedData(Plato::data::layout_t::SCALAR, size, initVal){}
};

class FauxParameter {
  public:
    FauxParameter(std::string name, std::string context, double value) : 
      m_name(name), m_context(context), m_value(value){}

    void setData(const std::vector<double> & aData)
    {
      m_value = aData[0];
    }
    void getData(std::vector<double> & aData) const
    {
      aData.resize(1);
      aData[0] = m_value;
    }
    int size() const
    {
      return 1;
    }

    std::string myContext() const {return m_context;}
    void setContext(std::string context) {m_context = context;}

    Plato::data::layout_t myLayout() const 
    {
      return Plato::data::layout_t::SCALAR_PARAMETER;
    }

  private:
    std::string m_name;
    std::string m_context;
    double m_value;
};
