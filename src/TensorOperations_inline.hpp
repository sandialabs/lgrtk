/*
 * TensorOperations_inline.hpp
 *
 *  Created on: Dec 12, 2017
 *      Author: swbova
 */

#ifndef SRC_TENSOROPERATIONS_INLINE_HPP_
#define SRC_TENSOROPERATIONS_INLINE_HPP_

#include "TensorOperations.hpp"
#include "FieldsEnum.hpp"
#include "Omega_h_few.hpp"

namespace lgr {

namespace tensorOps {

template <>
KOKKOS_INLINE_FUNCTION Scalar trace< Fields<3>::elem_sym_tensor_type >(int index,
        const  Fields<3>::elem_sym_tensor_type tensor) {

    return tensor(index, FieldsEnum<3>::K_S_XX) + tensor(index, FieldsEnum<3>::K_S_YY) +
            tensor(index,  FieldsEnum<3>::K_S_ZZ);
}
template <>
KOKKOS_INLINE_FUNCTION Scalar trace< Fields<2>::elem_sym_tensor_type >(int index,
        const Fields<2>::elem_sym_tensor_type tensor) {

    return tensor(index, FieldsEnum<2>::K_S_XX) + tensor(index, FieldsEnum<2>::K_S_YY);

}
template <>
KOKKOS_INLINE_FUNCTION Scalar trace< Fields<3>::elem_tensor_type >(int index,
        const  Fields<3>::elem_tensor_type tensor) {

    return tensor(index, FieldsEnum<3>::K_F_XX) + tensor(index, FieldsEnum<3>::K_F_YY) +
            tensor(index,  FieldsEnum<3>::K_F_ZZ);
}
template <>
KOKKOS_INLINE_FUNCTION Scalar trace< Fields<2>::elem_tensor_type >(int index,
        const Fields<2>::elem_tensor_type tensor) {

    return tensor(index, FieldsEnum<2>::K_F_XX) + tensor(index, FieldsEnum<2>::K_F_YY);

}

template <>
KOKKOS_INLINE_FUNCTION
Scalar trace< Fields<3>::elem_sym_tensor_state_type>(int index, int state,
        const Fields<3>::elem_sym_tensor_state_type tensor){

    return tensor(index, FieldsEnum<3>::K_S_XX, state) + tensor(index, FieldsEnum<3>::K_S_YY, state) +
            tensor(index,  FieldsEnum<3>::K_S_ZZ, state);
}
template <>
KOKKOS_INLINE_FUNCTION
Scalar trace< Fields<2>::elem_sym_tensor_state_type>(int index, int state,
        const Fields<2>::elem_sym_tensor_state_type tensor) {

    return tensor(index, FieldsEnum<2>::K_S_XX, state) + tensor(index, FieldsEnum<2>::K_S_YY, state);

}

template <>
KOKKOS_INLINE_FUNCTION  void
symmTimesVector<3>(const Scalar * symmTensor, const Scalar vx, const Scalar vy, const Scalar vz,
        Scalar * result){

    result[0] = symmTensor[FieldsEnum<3>::K_S_XX] * vx + symmTensor[FieldsEnum<3>::K_S_XY] * vy +
            symmTensor[FieldsEnum<3>::K_S_XZ] * vz;

    result[1] = symmTensor[FieldsEnum<3>::K_S_YX] * vx + symmTensor[FieldsEnum<3>::K_S_YY] * vy +
            symmTensor[FieldsEnum<3>::K_S_YZ] * vz;

    result[2] = symmTensor[FieldsEnum<3>::K_S_ZX] * vx + symmTensor[FieldsEnum<3>::K_S_ZY] * vy +
            symmTensor[FieldsEnum<3>::K_S_ZZ] * vz;
}
template <>
KOKKOS_INLINE_FUNCTION  void
symmTimesVector<2>(const Scalar * symmTensor, const Scalar vx, const Scalar vy, const Scalar,
        Scalar * result){

result[0] = symmTensor[FieldsEnum<2>::K_S_XX] * vx + symmTensor[FieldsEnum<3>::K_S_XY] * vy;
result[1] = symmTensor[FieldsEnum<2>::K_S_YX] * vx + symmTensor[FieldsEnum<3>::K_S_YY] * vy;

}

template <>
KOKKOS_INLINE_FUNCTION
Scalar dot<3>(const Scalar * a, const Scalar bx, const Scalar by, const Scalar bz) {
    return a[0]*bx + a[1]*by + a[2]*bz;
}
// return a dot b
template <>
KOKKOS_INLINE_FUNCTION
Scalar dot<2>(const Scalar * a, const Scalar bx, const Scalar by, const Scalar){
    return a[0]*bx + a[1]*by;
}

//result = 0.5*(oldTensorState + newTensorState)
template <>
KOKKOS_INLINE_FUNCTION
void avgTensorState(const int index, const int state0, const int state1,
        const Fields<3>::elem_sym_tensor_state_type tensor, Scalar * resultOfLength6 ) {

    resultOfLength6[FieldsEnum<3>::K_S_XX] =
            0.5 * (tensor(index, FieldsEnum<3>::K_S_XX, state0) + tensor(index, FieldsEnum<3>::K_S_XX, state1));
    resultOfLength6[FieldsEnum<3>::K_S_YY] =
            0.5 * (tensor(index, FieldsEnum<3>::K_S_YY, state0) + tensor(index, FieldsEnum<3>::K_S_YY, state1));
    resultOfLength6[FieldsEnum<3>::K_S_ZZ] =
            0.5 * (tensor(index, FieldsEnum<3>::K_S_ZZ, state0) + tensor(index, FieldsEnum<3>::K_S_ZZ, state1));
    resultOfLength6[FieldsEnum<3>::K_S_XY] =
            0.5 * (tensor(index, FieldsEnum<3>::K_S_XY, state0) + tensor(index, FieldsEnum<3>::K_S_XY, state1));
    resultOfLength6[FieldsEnum<3>::K_S_YZ] =
            0.5 * (tensor(index, FieldsEnum<3>::K_S_YZ, state0) + tensor(index, FieldsEnum<3>::K_S_YZ, state1));
    resultOfLength6[FieldsEnum<3>::K_S_ZX] =
            0.5 * (tensor(index, FieldsEnum<3>::K_S_ZX, state0) + tensor(index, FieldsEnum<3>::K_S_ZX, state1));


}

//result = 0.5*(oldTensorState + newTensorState)
template <>
KOKKOS_INLINE_FUNCTION
void avgTensorState(const int index, const int state0, const int state1,
        const Fields<2>::elem_sym_tensor_state_type tensor, Scalar * resultOfLength3 ){

    resultOfLength3[FieldsEnum<2>::K_S_XX] =
             0.5 * (tensor(index, FieldsEnum<3>::K_S_XX, state0) + tensor(index, FieldsEnum<3>::K_S_XX, state1));
     resultOfLength3[FieldsEnum<2>::K_S_YY] =
             0.5 * (tensor(index, FieldsEnum<3>::K_S_YY, state0) + tensor(index, FieldsEnum<3>::K_S_YY, state1));
     resultOfLength3[FieldsEnum<2>::K_S_XY] =
             0.5 * (tensor(index, FieldsEnum<3>::K_S_XY, state0) + tensor(index, FieldsEnum<3>::K_S_XY, state1));

}


template <>
KOKKOS_INLINE_FUNCTION
void avgVectors<Fields<3>::geom_array_type>(const int inputIndex, const int outputIndex,
        const Fields<3>::geom_array_type vector1, const Fields<3>::geom_array_type vector2,
        Scalar * result_x, Scalar * result_y, Scalar * result_z ){

    result_x[outputIndex] = 0.5 * (vector1(inputIndex, 0) + vector2(inputIndex, 0));
    result_y[outputIndex] = 0.5 * (vector1(inputIndex, 1) + vector2(inputIndex, 1));
    result_z[outputIndex] = 0.5 * (vector1(inputIndex, 2) + vector2(inputIndex, 2));

}
template <>
KOKKOS_INLINE_FUNCTION
void avgVectors<Fields<2>::geom_array_type>(const int inputIndex, const int outputIndex,
        const Fields<2>::geom_array_type vector1, const Fields<2>::geom_array_type vector2,
        Scalar * result_x, Scalar * result_y, Scalar *){

    result_x[outputIndex] = 0.5 * (vector1(inputIndex, 0) + vector2(inputIndex, 0));
    result_y[outputIndex] = 0.5 * (vector1(inputIndex, 1) + vector2(inputIndex, 1));

}

// x = a*x + y + c*I
template <>
KOKKOS_INLINE_FUNCTION
void symm_axpypc<3>(const Scalar a, Scalar * x, const Scalar * y, const Scalar c){
    x[FieldsEnum<3>::K_S_XX] = a*x[FieldsEnum<3>::K_S_XX] + c + y[FieldsEnum<3>::K_S_XX];
    x[FieldsEnum<3>::K_S_YY] = a*x[FieldsEnum<3>::K_S_YY] + c + y[FieldsEnum<3>::K_S_YY];
    x[FieldsEnum<3>::K_S_ZZ] = a*x[FieldsEnum<3>::K_S_ZZ] + c + y[FieldsEnum<3>::K_S_ZZ];

    x[FieldsEnum<3>::K_S_XY] = a*x[FieldsEnum<3>::K_S_XY] + y[FieldsEnum<3>::K_S_XY];
    x[FieldsEnum<3>::K_S_YZ] = a*x[FieldsEnum<3>::K_S_YZ] + y[FieldsEnum<3>::K_S_YZ];
    x[FieldsEnum<3>::K_S_ZX] = a*x[FieldsEnum<3>::K_S_ZX] + y[FieldsEnum<3>::K_S_ZX];



}

template <>
KOKKOS_INLINE_FUNCTION
void symm_axpypc<2>(const Scalar a, Scalar * x, const Scalar * y, const Scalar c) {
    x[FieldsEnum<3>::K_S_XX] = a*x[FieldsEnum<3>::K_S_XX] + c + y[FieldsEnum<3>::K_S_XX];
    x[FieldsEnum<3>::K_S_YY] = a*x[FieldsEnum<3>::K_S_YY] + c + y[FieldsEnum<3>::K_S_YY];

    x[FieldsEnum<3>::K_S_XY] = a*x[FieldsEnum<3>::K_S_XY] + y[FieldsEnum<3>::K_S_XY];

}

template <>
KOKKOS_INLINE_FUNCTION
void initDiagonal< Fields<3>::elem_sym_tensor_state_type>(const int index,
        const int state, const Scalar a, Fields<3>::elem_sym_tensor_state_type tensor) {

    tensor(index, FieldsEnum<3>::K_S_XX, state) = a;
    tensor(index, FieldsEnum<3>::K_S_YY, state) = a;
    tensor(index, FieldsEnum<3>::K_S_ZZ, state) = a;

}
template <>
KOKKOS_INLINE_FUNCTION
void initDiagonal< Fields<2>::elem_sym_tensor_state_type>(const int index, const int state,
        const Scalar a,Fields<2>::elem_sym_tensor_state_type tensor){

    tensor(index, FieldsEnum<2>::K_S_XX, state) = a;
    tensor(index, FieldsEnum<2>::K_S_YY, state) = a;
}
template <>
KOKKOS_INLINE_FUNCTION
Omega_h::Matrix<3,3> fillMatrix<Omega_h::Matrix<3,3> >(const Fields<3>::elem_tensor_type tensor, const int entityIndex) {
    Omega_h::Matrix<3,3> matrixToFill;
    matrixToFill(0, 0) = tensor(entityIndex, FieldsEnum<3>::K_F_XX);
    matrixToFill(0, 1) = tensor(entityIndex, FieldsEnum<3>::K_F_XY);
    matrixToFill(0, 2) = tensor(entityIndex, FieldsEnum<3>::K_F_XZ);

    matrixToFill(1, 0) = tensor(entityIndex, FieldsEnum<3>::K_F_YX);
    matrixToFill(1, 1) = tensor(entityIndex, FieldsEnum<3>::K_F_YY);
    matrixToFill(1, 2) = tensor(entityIndex, FieldsEnum<3>::K_F_YZ);

    matrixToFill(2, 0) = tensor(entityIndex, FieldsEnum<3>::K_F_ZX);
    matrixToFill(2, 1) = tensor(entityIndex, FieldsEnum<3>::K_F_ZY);
    matrixToFill(2, 2) = tensor(entityIndex, FieldsEnum<3>::K_F_ZZ);
    return matrixToFill;
}

template <>
KOKKOS_INLINE_FUNCTION
Omega_h::Matrix<2,2> fillMatrix<Omega_h::Matrix<2,2>>(const Fields<2>::elem_tensor_type tensor, const int entityIndex){

    Omega_h::Matrix<2,2> matrixToFill;
    matrixToFill(0, 0) = tensor(entityIndex, FieldsEnum<2>::K_F_XX);
    matrixToFill(0, 1) = tensor(entityIndex, FieldsEnum<2>::K_F_XY);

    matrixToFill(1, 0) = tensor(entityIndex, FieldsEnum<2>::K_F_YX);
    matrixToFill(1, 1) = tensor(entityIndex, FieldsEnum<2>::K_F_YY);
    return matrixToFill;
}


}

}


#endif /* SRC_TENSOROPERATIONS_INLINE_HPP_ */
