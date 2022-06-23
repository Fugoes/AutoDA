#include "dag.hpp"

void test_simple() {
  autoda::Dag dag(3, 2);
  dag.append(autoda::ops::SS_S_ADD, 0, 1);
  dag.append(autoda::ops::SS_S_MUL, 0, 5);
  dag.append(autoda::ops::VS_V_ADD, 3, 2);
  dag.append(autoda::ops::VV_V_ADD, 4, 7);
  dag.append(autoda::ops::VS_V_MUL, 8, 6);
  dag.append(autoda::ops::VV_V_DIV, 7, 8);
  dag.display(std::cout);
  dag.type_check();

  std::cout << '\n';

  autoda::TacProgram program{};
  fstd::dynarray<uint8_t> S_output(0), V_output(1);
  dag.compile<false>(program, {}, {9}, S_output, V_output);
  program.display(std::cout);

  std::cout << '\n';
}

void test_input() {
  autoda::Dag dag(3, 2);
  dag.append(autoda::ops::SS_S_ADD, 0, 1);
  dag.append(autoda::ops::SS_S_MUL, 0, 5);
  dag.append(autoda::ops::VS_V_ADD, 4, 2);
  dag.append(autoda::ops::VV_V_ADD, 4, 7);
  dag.append(autoda::ops::VS_V_MUL, 8, 6);
  dag.append(autoda::ops::VV_V_DIV, 7, 8);
  dag.display(std::cout);
  dag.type_check();

  std::cout << '\n';

  autoda::TacProgram program{};
  fstd::dynarray<uint8_t> S_output(3), V_output(2);
  dag.compile<false>(program, {0, 1, 2}, {3, 10}, S_output, V_output);
  program.display(std::cout);

  std::cout << '\n';
}

void test_copy() {
  autoda::Dag dag(3, 2);
  dag.display(std::cout);
  dag.type_check();

  std::cout << '\n';

  autoda::TacProgram program{};
  fstd::dynarray<uint8_t> S_output(3), V_output(2);
  dag.compile<true>(program, {1, 2, 0}, {4, 3}, S_output, V_output);
  program.display(std::cout);

  std::cout << '\n';
}

void test_no_copy() {
  autoda::Dag dag(3, 2);
  dag.display(std::cout);
  dag.type_check();

  std::cout << '\n';

  autoda::TacProgram program{};
  fstd::dynarray<uint8_t> S_output(3), V_output(2);
  dag.compile<true>(program, {0, 1, 2}, {3, 4}, S_output, V_output);
  program.display(std::cout);

  std::cout << '\n';
}

int main() {
  std::cout << "test_simple()\n";
  test_simple();
  std::cout << "test_input()\n";
  test_input();
  std::cout << "test_copy()\n";
  test_copy();
  std::cout << "test_no_copy()\n";
  test_no_copy();

  return 0;
}