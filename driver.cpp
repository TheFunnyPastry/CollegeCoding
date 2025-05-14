// driver.cpp
// Bob Myers
//
// Test the Add and Equals functions

#include <iostream>		// for cout
#include "frac.h"		// for Fraction declarations

using namespace std;

int main()
{
   Fraction f1, f2, f3, f4;
   cout << "Enter fraction f1: ";
   f1.Input();
   cout << "Enter fraction f2: ";
   f2.Input();
   
   cout << "\nf1 = ";
   f1.Show();
   cout << "\nf2 = ";
   f2.Show();

   cout << "\n\n";

   if (Equals(f1, f2))
	cout << "f1 and f2 are equal\n";
   else
	cout << "f1 and f2 are NOT equal\n";

   f3 = Add(f1, f2);
   cout << "f1 + f2 = ";
   f3.Show();
   cout << '\n';

   return 0;
}