//--------------- MAIN.CPP ---------------

// Driver routine to test SOME of the functions of the Fraction class 

#include <iostream>		// for cout
#include "frac.h"		// for Fraction declarations

using namespace std;

int main()
{
	// Try all three possible fraction constructors
	// and the input/output routines.
	
	// These declarations use the default constructor
	Fraction f1, f2;

	// These declarations use the constructor with parameters
	Fraction f3(3,4), f4(6);

	// Use the objects

	cout << "\n The fraction f1 is ";
	f1.Show();
	
	cout << "\n The fraction f2 is ";
	f2.Show();

	cout << "\n The fraction f3 is ";
	f3.Show();

	cout << "\n The fraction f4 is ";
	f4.Show();
	
	cout << "\n Now enter first fraction: ";
	f1.Input();
	cout << "\nYou entered ";
	f1.Show();

	cout << "\n Now enter second fraction: ";
	f2.Input();
	cout << "\nYou entered ";
	f2.Show();

	// Finally, find the floating-point value of f1 and f2
	
	cout << "\n The value of fraction 1 is " << f1.Evaluate() << '\n';
	cout << "\n The value of fraction 2 is " << f2.Evaluate() << '\n';

	cout << "Goodbye!\n";
}
