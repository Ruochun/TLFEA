// Simple test to validate namespace structure
#include "src/types.h"
#include <iostream>

int main() {
    // Test that tlfea types are accessible
    tlfea::Real x = 3.14;
    tlfea::VectorXR vec(3);
    vec << 1.0, 2.0, 3.0;
    
    tlfea::MatrixXR mat(2, 2);
    mat << 1.0, 2.0, 3.0, 4.0;
    
    std::cout << "tlfea namespace test successful!" << std::endl;
    std::cout << "Real value: " << x << std::endl;
    std::cout << "Vector: " << vec.transpose() << std::endl;
    std::cout << "Matrix:\n" << mat << std::endl;
    
    return 0;
}
