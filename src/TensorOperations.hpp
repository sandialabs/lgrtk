/*
 * TensorOperations.hpp
 *
 *  Created on: Dec 12, 2017
 *      Author: swbova
 */

#ifndef SRC_TENSOROPERATIONS_HPP_
#define SRC_TENSOROPERATIONS_HPP_

#include "Omega_h_few.hpp"

#include "Fields.hpp"


namespace lgr {

namespace tensorOps {

// return trace( tensor(index) )
template <typename FieldView>
KOKKOS_INLINE_FUNCTION
Scalar trace(int index, const FieldView tensor);

template <typename FieldView>
KOKKOS_INLINE_FUNCTION
Scalar trace(int index, int state, const FieldView tensor);


// result =  symmTensor*v
template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void symmTimesVector(const Scalar * symmTensor, const Scalar vx, const Scalar vy, const Scalar vz,
        Scalar * result);

// return a dot b
template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar dot(const Scalar * a, const Scalar bx, const Scalar by, const Scalar bz);

//result = 0.5*(oldTensorState + newTensorState)
template <typename FieldView>
KOKKOS_INLINE_FUNCTION
void avgTensorState(const int index, const int state0, const int state1,
        const FieldView tensorWithState, Scalar * result );

//result[outputIndex] = 0.5*(vector1[inputIndex] + vector2[inputIndex])
template <typename FieldView>
KOKKOS_INLINE_FUNCTION
void avgVectors(const int inputIndex, const int outputIndex,
        const FieldView vector1,const FieldView vector2,  Scalar * result_x, Scalar * result_y, Scalar * result_z );

// x = a*x + y + constant*I, x,y symm tensors
template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void symm_axpypc(const Scalar a, Scalar * x, const Scalar * y, const Scalar c);

// tensor = a*I
template <typename View>
KOKKOS_INLINE_FUNCTION
void initDiagonal(const int index, const int state, const Scalar a, View tensor);

template <typename MatrixT, typename View>
KOKKOS_INLINE_FUNCTION
MatrixT fillMatrix(const View tensor, const int entityIndex);

//------------------------------------------------------
template <>
KOKKOS_INLINE_FUNCTION
Scalar trace< Fields<3>::elem_sym_tensor_type >(int index,
        const  Fields<3>::elem_sym_tensor_type  tensor);

template <>
KOKKOS_INLINE_FUNCTION
Scalar trace<Fields<2>::elem_sym_tensor_type >(int index,
        const Fields<2>::elem_sym_tensor_type tensor);
template <>
KOKKOS_INLINE_FUNCTION
Scalar trace< Fields<3>::elem_tensor_type >(int index,
        const  Fields<3>::elem_tensor_type  tensor);

template <>
KOKKOS_INLINE_FUNCTION
Scalar trace<Fields<2>::elem_tensor_type >(int index,
        const Fields<2>::elem_tensor_type tensor);


template <>
KOKKOS_INLINE_FUNCTION
Scalar trace< Fields<3>::elem_sym_tensor_state_type>(int index, int state,
        const Fields<3>::elem_sym_tensor_state_type tensor);
template <>
KOKKOS_INLINE_FUNCTION
Scalar trace< Fields<2>::elem_sym_tensor_state_type>(int index, int state,
        const Fields<2>::elem_sym_tensor_state_type tensor);

template <>
KOKKOS_INLINE_FUNCTION void
symmTimesVector<3>(const Scalar * symmTensor, const Scalar vx, const Scalar vy, const Scalar vz,
        Scalar * result );

template <>
KOKKOS_INLINE_FUNCTION void
symmTimesVector<2>(const Scalar * symmTensor, const Scalar vx, const Scalar vy, const Scalar vz,
        Scalar * result );

// return a dot b
template <>
KOKKOS_INLINE_FUNCTION
Scalar dot<3>(const Scalar * a, const Scalar bx, const Scalar by, const Scalar bz);
// return a dot b
template <>
KOKKOS_INLINE_FUNCTION
Scalar dot<2>(const Scalar * a, const Scalar bx, const Scalar by, const Scalar bz);

//result = 0.5*(oldTensorState + newTensorState)
template <>
KOKKOS_INLINE_FUNCTION
void avgTensorState(const int index, const int state0, const int state1,
        const Fields<3>::elem_sym_tensor_state_type tensor, Scalar * resultOfLength6 );

//result = 0.5*(oldTensorState + newTensorState)
template <>
KOKKOS_INLINE_FUNCTION
void avgTensorState(const int index, const int state0, const int state1,
        const Fields<3>::elem_sym_tensor_state_type tensor, Scalar * resultOfLength3 );


template <>
KOKKOS_INLINE_FUNCTION
void avgVectors<Fields<3>::geom_array_type>(const int inputIndex, const int outputIndex,
        const Fields<3>::geom_array_type vector1,const Fields<3>::geom_array_type vector2 , Scalar * result_x, Scalar * result_y, Scalar * result_z );
template <>
KOKKOS_INLINE_FUNCTION
void avgVectors<Fields<2>::geom_array_type>(const int inputIndex, const int outputIndex,
        const Fields<2>::geom_array_type vector1,const Fields<2>::geom_array_type vector2 , Scalar * result_x, Scalar * result_y, Scalar * result_z );

// x = ax + y + constant
template <>
KOKKOS_INLINE_FUNCTION
void symm_axpypc<3>(const Scalar a, Scalar * x, const Scalar * y, const Scalar c);

template <>
KOKKOS_INLINE_FUNCTION
void symm_axpypc<2>(const Scalar a, Scalar * x, const Scalar * y, const Scalar c);

template <>
KOKKOS_INLINE_FUNCTION
void initDiagonal< Fields<3>::elem_sym_tensor_state_type>(const int index, const int state,
        const Scalar a,Fields<3>::elem_sym_tensor_state_type tensor);
template <>
KOKKOS_INLINE_FUNCTION
void initDiagonal< Fields<2>::elem_sym_tensor_state_type>(const int index, const int state,
        const Scalar a,Fields<2>::elem_sym_tensor_state_type tensor);

template <>
KOKKOS_INLINE_FUNCTION
Omega_h::Matrix<3,3> fillMatrix<Omega_h::Matrix<3,3> >(const Fields<3>::elem_tensor_type tensor, const int entityIndex);

template <>
KOKKOS_INLINE_FUNCTION
Omega_h::Matrix<2,2> fillMatrix<Omega_h::Matrix<2,2>> (const Fields<2>::elem_tensor_type tensor, const int entityIndex);


}
}


#endif /* SRC_TENSOROPERATIONS_HPP_ */
