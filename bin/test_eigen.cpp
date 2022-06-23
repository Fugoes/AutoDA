#include <iostream>
#include <eigen3/Eigen/Dense>

int main() {
  Eigen::Array<float, 4, Eigen::Dynamic> arr(4, 10);
  for (unsigned i = 0; i < 4; i++)
    for (unsigned j = 0; j < 10; j++) {
      arr(i, j) = (i * j + 1) / 10.0;
    }
  std::cout << arr << std::endl;

  printf("arr.col(0) = arr.col(0) + arr.col(1)\n");
  arr.col(0) = arr.col(0) + arr.col(1);
  std::cout << arr << std::endl;

  printf("arr.col(0) = arr.col(0) - arr.col(1)\n");
  arr.col(0) = arr.col(0) - arr.col(1);
  std::cout << arr << std::endl;

  printf("arr.col(0) = arr.col(0) * arr.col(1)\n");
  arr.col(0) = arr.col(0) * arr.col(1);
  std::cout << arr << std::endl;

  printf("arr.col(0) = arr.col(0) / arr.col(1)\n");
  arr.col(0) = arr.col(0) / arr.col(1);
  std::cout << arr << std::endl;

  printf("arr.col(0) = arr.col(0) + 0.9\n");
  arr.col(0) = arr.col(0) + 0.9;
  std::cout << arr << std::endl;

  printf("arr.col(1).matrix().dot(arr.col(0).matrix())\n");
  std::cout << arr.col(1).matrix().dot(arr.col(0).matrix()) << std::endl;

  arr.col(1) = arr.col(1).square();
  std::cout << arr << std::endl;

  std::cout << arr.col(0).matrix().norm() << std::endl;

  arr.col(0).fill(0.0);
  std::cout << arr << std::endl;

  arr.col(0) = arr.col(1) / arr.col(0);
  std::cout << arr << std::endl;

  arr.col(0) = 1 / arr.col(0);
  std::cout << arr << std::endl;

  return 0;
}