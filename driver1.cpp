/////////////////////////////////////////////////////////
// Test program for the Date class
/////////////////////////////////////////////////////////
#include <iostream>
#include "date.h"
using namespace std;
int main()
{
Date d1; // should default to 1/1/2025
Date d2(2,28,1980); // should init to 2/28/1992
// display dates to the screen
cout << "\nDate d1 is: ";
d1.ShowByDay();
cout << "\nDate d2 is: ";
d2.ShowByDay();
d2.Increment();
cout << "\nAfter increment, Date d2 is: ";
d2.ShowByDay();
cout << "\nShown by month, Date d2 is: \n\n";
d2.ShowByMonth();
cout << endl;
int ret = d1.Compare(d2);
if(ret == -1)
cout << "d1 is before d2" << endl;
else if (ret == 1)
cout << "d1 is after d2" << endl;
else
cout << "d1 is equal to d2" << endl;
}
