#include "ops.hpp"

namespace {

const uint8_t name_table[] = {
#define X(sig, name, code, t1, t2, t0) 0,
    OPS_BINARY
#undef X
#define X(sig, name, code, t1, t0) 0,
    OPS_UNARY
#undef X
#define X(sig, name, code, t1, t0) 0,
    OPS_COPY
#undef X
#define X(sig, name, code, t0) 0,
    OPS_INPUT
#undef X
};
static_assert(sizeof(name_table) == OPS_COUNT * sizeof(uint8_t), "");

const uint8_t output_type_table[] = {
#define X(sig, name, code, t1, t2, t0) 0,
    OPS_BINARY
#undef X
#define X(sig, name, code, t1, t0) 0,
    OPS_UNARY
#undef X
#define X(sig, name, code, t1, t0) 0,
    OPS_COPY
#undef X
#define X(sig, name, code, t0) 0,
    OPS_INPUT
#undef X
};
static_assert(sizeof(output_type_table) == OPS_COUNT * sizeof(uint8_t), "");

const uint8_t input_type_1_table[] = {
#define X(sig, name, code, t1, t2, t0) 0,
    OPS_BINARY
#undef X
#define X(sig, name, code, t1, t0) 0,
    OPS_UNARY
#undef X
#define X(sig, name, code, t1, t0) 0,
    OPS_COPY
#undef X
};
static_assert(sizeof(input_type_1_table) == OPS_TAC_COUNT * sizeof(uint8_t), "");

const uint8_t input_type_2_table[] = {
#define X(sig, name, code, t1, t2, t0) 0,
    OPS_BINARY
#undef X
};
static_assert(sizeof(input_type_2_table) == OPS_BINARY_COUNT * sizeof(uint8_t), "");

};

namespace autoda {
namespace ops {

const char *name(uint8_t op) {
  switch (op) {
#define X(sig, name, code, t1, t2, t0) case sig ##_## name: return #name;
    OPS_BINARY
#undef X
#define X(sig, name, code, t1, t0) case sig ##_## name: return #name;
    OPS_UNARY
#undef X
#define X(sig, name, code, t1, t0) case sig ##_## name: return #name;
    OPS_COPY
#undef X
#define X(sig, name, code, t0) case sig ##_## name: return #name;
    OPS_INPUT
#undef X
    default: {
      bug();
      return "";
    }
  }
}

const char *fullname(uint8_t op) {
  switch (op) {
#define X(sig, name, code, t1, t2, t0) case sig ##_## name: return #sig "_" #name;
    OPS_BINARY
#undef X
#define X(sig, name, code, t1, t0) case sig ##_## name: return #sig "_" #name;
    OPS_UNARY
#undef X
#define X(sig, name, code, t1, t0) case sig ##_## name: return #sig "_" #name;
    OPS_COPY
#undef X
#define X(sig, name, code, t0) case sig ##_## name: return #sig "_" #name;
    OPS_INPUT
#undef X
    default: {
      bug();
      return "";
    }
  }
}

uint16_t number_of_input(uint8_t op) {
  switch (op) {
#define X(sig, name, code, t1, t2, t0) case sig ##_## name: return 2;
    OPS_BINARY
#undef X
#define X(sig, name, code, t1, t0) case sig ##_## name: return 1;
    OPS_UNARY
#undef X
#define X(sig, name, code, t1, t0) case sig ##_## name: return 1;
    OPS_COPY
#undef X
#define X(sig, name, code, t0) case sig ##_## name: return 0;
    OPS_INPUT
#undef X
    default: {
      bug();
      return UINT16_MAX;
    }
  }
}

DataType output_type(uint8_t op) {
  switch (op) {
#define X(sig, name, code, t1, t2, t0) case sig ##_## name: return DataType::t0;
    OPS_BINARY
#undef X
#define X(sig, name, code, t1, t0) case sig ##_## name: return DataType::t0;
    OPS_UNARY
#undef X
#define X(sig, name, code, t1, t0) case sig ##_## name: return DataType::t0;
    OPS_COPY
#undef X
#define X(sig, name, code, t0) case sig ##_## name: return DataType::t0;
    OPS_INPUT
#undef X
    default: {
      bug();
      return DataType::Scalar;
    }
  }
}

DataType input_type_1(uint8_t op) {
  switch (op) {
#define X(sig, name, code, t1, t2, t0) case sig ##_## name: return DataType::t1;
    OPS_BINARY
#undef X
#define X(sig, name, code, t1, t0) case sig ##_## name: return DataType::t1;
    OPS_UNARY
#undef X
#define X(sig, name, code, t1, t0) case sig ##_## name: return DataType::t1;
    OPS_COPY
#undef X
    default: {
      bug();
      return DataType::Scalar;
    }
  }
}

DataType input_type_2(uint8_t op) {
  switch (op) {
#define X(sig, name, code, t1, t2, t0) case sig ##_## name: return DataType::t2;
    OPS_BINARY
#undef X
    default: {
      bug();
      return DataType::Scalar;
    }
  }
}

const char *prefix_of(DataType dt) {
  if (dt == DataType::Scalar) {
    return "s";
  } else { // dt == DataType::Vector
    return "v";
  }
}

}
}
