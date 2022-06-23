#include "dag.hpp"

int main() {
  {
    autoda::Dag dag(2, 3);
    dag.append(autoda::ops::SS_S_ADD, 0, 1);
    dag.append(autoda::ops::S_S_SQRT_ABS, 5);
    dag.append(autoda::ops::VS_V_DIV, 2, 6);
    dag.display(std::cout);
    std::cout << '\n';
  }

  // {
  //   auto dag = autoda::Dag::random(2, 3, 50);
  //   std::vector<uint16_t> roots{};
  //   roots.push_back((uint16_t) dag.nodes_.size() - 1);
  //   uint16_t root_max = 0;
  //   dag.gc_mark_roots(roots, root_max);
  //   dag.gc_mark(root_max);
  //   dag.display(std::cout);
  // }

  return 0;
}