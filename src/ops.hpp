#ifndef AUTODA_OPS_HPP
#define AUTODA_OPS_HPP

#include "prelude.hpp"

// S for scalar. V for vector. It seems that we do not need matrix operations here? The SS_S_
// prefix indicates the operation's operands are two scalars and it returns a scalar.

#define OPS_COUNT 29
#define OPS_TAC_COUNT 27
#define OPS_BINARY_COUNT 17

#define OPS_BINARY                                \
  X(SS_S, ADD,         0, Scalar, Scalar, Scalar) \
  X(SS_S, SUB,         1, Scalar, Scalar, Scalar) \
  X(SS_S, MUL,         2, Scalar, Scalar, Scalar) \
  X(SS_S, DIV,         3, Scalar, Scalar, Scalar) \
  X(VV_V, ADD,         4, Vector, Vector, Vector) \
  X(VV_V, SUB,         5, Vector, Vector, Vector) \
  X(VV_V, MUL,         6, Vector, Vector, Vector) \
  X(VV_V, DIV,         7, Vector, Vector, Vector) \
  X(VS_V, ADD,         8, Vector, Scalar, Vector) \
  X(VS_V, SUB,         9, Vector, Scalar, Vector) \
  X(SV_V, SUB,        10, Scalar, Vector, Vector) \
  X(VS_V, MUL,        11, Vector, Scalar, Vector) \
  X(VS_V, DIV,        12, Vector, Scalar, Vector) \
  X(SV_V, DIV,        13, Scalar, Vector, Vector) \
  X(VV_S, DOT,        14, Vector, Vector, Scalar) \
  X(SS_V, NORMAL,     15, Scalar, Scalar, Vector) \
  X(SS_V, UNIFORM,    16, Scalar, Scalar, Vector)
#define OPS_UNARY                                 \
  X(S_S,  SQUARE,     17, Scalar, Scalar)         \
  X(V_V,  SQUARE,     18, Vector, Vector)         \
  X(V_S,  NORM,       19, Vector, Scalar)         \
  X(V_S,  SUM,        20, Vector, Scalar)         \
  X(V_V,  SQRT_ABS,   21, Vector, Vector)         \
  X(S_S,  SQRT_ABS,   22, Scalar, Scalar)         \
  X(S_S,  RELU,       23, Scalar, Scalar)         \
  X(V_V,  RELU,       24, Vector, Vector)
#define OPS_COPY                                  \
  X(S_S,  COPY,       25, Scalar, Scalar)         \
  X(V_V,  COPY,       26, Vector, Vector)
#define OPS_INPUT                                 \
  X(S,    INPUT,      27, Scalar)                 \
  X(V,    INPUT,      28, Vector)

namespace autoda {
namespace ops {

enum class DataType { Scalar, Vector };

#define X(sig, name, code, t1, t2, t0) static const uint8_t sig ##_## name = code;
OPS_BINARY
#undef X
#define X(sig, name, code, t1, t0) static const uint8_t sig ##_## name = code;
OPS_UNARY
#undef X
#define X(sig, name, code, t1, t0) static const uint8_t sig ##_## name = code;
OPS_COPY
#undef X
#define X(sig, name, code, t0) static const uint8_t sig ##_## name = code;
OPS_INPUT
#undef X

const char *name(uint8_t op);

const char *fullname(uint8_t op);

DataType output_type(uint8_t op);

uint16_t number_of_input(uint8_t op);

DataType input_type_1(uint8_t op);

DataType input_type_2(uint8_t op);

const char *prefix_of(DataType dt);

}
}

#endif //AUTODA_OPS_HPP
