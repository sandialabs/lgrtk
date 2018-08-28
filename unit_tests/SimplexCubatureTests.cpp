#include "Teuchos_UnitTestHarness.hpp"

#include "LGRLambda.hpp"
#include "Cubature.hpp"

using namespace lgr;

namespace {
  typedef double Scalar;
  typedef Kokkos::DefaultExecutionSpace MemSpace;
  
  typedef Kokkos::View<Scalar**, Layout, MemSpace>      RefPointsView;     // (P,D)
  typedef Kokkos::View<Scalar*,  MemSpace>      WeightsView;       // (P)
  
  void testVolumeRecovery(int spaceDim, int maxPolyOrder, Teuchos::FancyOStream &out, bool &success)
  {
    /*
     
     Our simplex nodes on the reference cell are:
      - along each coordinate axis, at unit distance from origin
      - the origin
     
     So in 1D the "volume" of the reference cell is 1.
     In 2D, the volume is 1/2
     In 3D, the volume is 1/6
     For n dimensions, the volume is 1/(n!)
     
     (See the "Volume" section of: https://en.wikipedia.org/wiki/Simplex )
     
     In this test, we integrate the unit function on the reference cell, and confirm
     that this matches the volume.  We do this for a range of cubature degrees, 
     from 0 to maxPolyOrder.
     
     */
    double expectedVolume = 1.0;
    for (int d=0; d<spaceDim; d++)
    {
      expectedVolume *= 1.0 / double(d+1);
    }
    
    using namespace std;
    
    double tol = 1e-15; // we expect to get this pretty accurately...
    for (int polyOrder=0; polyOrder<=maxPolyOrder; polyOrder++)
    {
      int numPoints = Cubature::getNumCubaturePoints(spaceDim,polyOrder);
      
      RefPointsView points("points",numPoints,spaceDim);
      WeightsView   weights("weights",numPoints);
      
/*      cout << "points.dimension(0): " << points.dimension(0) << endl;
      cout << "points.dimension(1): " << points.dimension(1) << endl;
      cout << "points.size(): " << points.size() << endl;
      cout << "weights.dimension(0): " << weights.dimension(0) << endl;
      cout << "weights.size(): " << weights.size() << endl;*/
      
      Cubature::getCubature(spaceDim,polyOrder,points,weights);
      double volume = 0.0;
      /*for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        volume += weights(ptOrdinal);
      }*/
      Kokkos::parallel_reduce(numPoints, LAMBDA_EXPRESSION(const int ptOrdinal, double &localSum) {
        localSum += weights(ptOrdinal);
      }, volume);
      TEST_FLOATING_EQUALITY(volume, expectedVolume, tol);
    }
  }
  
  TEUCHOS_UNIT_TEST( Cubature, VolumeRecovery1D )
  {
    int spaceDim = 1;
    int maxPolyOrder = 3; // at present, we support up to exact cubic integration in 1-3D
    testVolumeRecovery(spaceDim,maxPolyOrder,out,success);
  }
  
  TEUCHOS_UNIT_TEST( Cubature, VolumeRecovery2D )
  {
    int spaceDim = 2;
    int maxPolyOrder = 3; // at present, we support up to exact cubic integration in 1-3D
    testVolumeRecovery(spaceDim,maxPolyOrder,out,success);
  }
  
  TEUCHOS_UNIT_TEST( Cubature, VolumeRecovery3D )
  {
    int spaceDim = 3;
    int maxPolyOrder = 3; // at present, we support up to exact cubic integration in 1-3D
    testVolumeRecovery(spaceDim,maxPolyOrder,out,success);
  }
} // namespace
