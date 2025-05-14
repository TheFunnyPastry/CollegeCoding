#include <iostream>
#include "polynomial.h"

using namespace std;

int main() {
    // Test default constructor
    Polynomial pDefault;
    cout << "Default polynomial: " << pDefault << endl; // Expected: "0" or "Polynomial of degree -1"
    
    // Test parameterized constructor
    Polynomial pParam(5);
    cout << "Parameterized polynomial (constructed with 5): " << pParam << endl;
    
    // Test setter functions
    pParam.setCoefficient(2, 10);  // set coefficient of x^2 to 10
    pParam.setLetter('m');          // change variable to 'm'
    cout << "After setCoefficient(2, 10) and setLetter('m'): " << pParam << endl;
    
    // Test arithmetic operator (addition) with one case
    Polynomial pA(3);  // creates a polynomial from integer 3
    Polynomial pB(2);  // creates a polynomial from integer 2
    Polynomial pSum = pA + pB;
    cout << "\nArithmetic Test (Addition):" << endl;
    cout << "pA: " << pA << endl;
    cout << "pB: " << pB << endl;
    cout << "pA + pB: " << pSum << endl;
    
    return 0;
}
