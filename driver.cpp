#include <iostream>
#include "sphere.h"
using namespace std;
void printSummaries(Sphere Sphere1, Sphere Sphere2, Sphere Sphere3, Sphere Sphere4,
Sphere Sphere5, int precision= 2);
int main() {
// seed the random number generator
srand((time(0)));
int precision = 0; // Precision variable for summary testing
double growAmount = 2.5; // Tracking variable for grow
double shrinkAmount = 2.0; // Tracking variable for shrink
// Create 5 Sphere objects
Sphere Sphere1, Sphere2, Sphere3, Sphere4, Sphere5;
Sphere1 = Sphere();
Sphere2 = Sphere(2.5);
Sphere3 = Sphere(1.54);
Sphere4 = Sphere(5.0);
Sphere5 = Sphere(32.32);
precision = 5;
printSummaries(Sphere1, Sphere2, Sphere3, Sphere4, Sphere5, precision);
cout<< "--> Growing Spheres by " << growAmount << endl;
Sphere1.grow(growAmount);
Sphere2.grow(growAmount);
Sphere3.grow(growAmount);
Sphere4.grow(growAmount);
Sphere5.grow(growAmount);
precision = 2;
printSummaries(Sphere1, Sphere2, Sphere3, Sphere4, Sphere5, precision);
cout<< "--> Shrinking Spheres by " << shrinkAmount<< endl;
Sphere1.shrink(shrinkAmount);
Sphere2.shrink(shrinkAmount);
Sphere3.shrink(shrinkAmount);
Sphere4.shrink(shrinkAmount);
Sphere5.shrink(shrinkAmount);
printSummaries(Sphere1, Sphere2, Sphere3, Sphere4, Sphere5);
return 0;
}
/*
* This function receives five Spheres and prints their summaries
*/
void printSummaries(Sphere Sphere1, Sphere Sphere2, Sphere Sphere3, Sphere Sphere4,
Sphere Sphere5, int precision){
cout << "Print summaries" << endl;
cout<< "Summary Sphere " << 1 << ""<< endl;
Sphere1.printSummary(precision);
cout << endl;
cout<< "Summary Sphere " << 2 << ""<< endl;
Sphere2.printSummary(precision);
cout << endl;
cout<< "Summary Sphere " << 3 << ""<< endl;
Sphere3.printSummary(precision);
cout << endl;
cout<< "Summary Sphere " << 4 << ""<< endl;
Sphere4.printSummary(precision);
cout << endl;
cout<< "Summary Sphere " << 5 << ""<< endl;
Sphere5.printSummary(precision);
cout << endl;
}
