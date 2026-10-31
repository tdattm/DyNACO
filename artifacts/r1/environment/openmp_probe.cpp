#include <omp.h>
#include <iostream>
int main() {
    omp_set_dynamic(0);
    omp_set_num_threads(2);
    int count = 0;
#pragma omp parallel reduction(+:count)
    count += 1;
    std::cout << "OpenMP=" << _OPENMP << " threads=" << count << '\n';
    return count == 2 ? 0 : 1;
}
