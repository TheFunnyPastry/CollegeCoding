#include <iostream>

using namespace std;


int list[10]; //delcares an array of size 10(0-9)
//cout << list[5];// 

int * p;// declaring a pointer(called p)  -- a "pointer-to-int"
cout << *p;// dereferecing for an existing pointer 

int x,y,z;//build three ints
//all the same declara a pointer
int* p;
int *p;
int * p;
int *p, *q, *r;// this is three pointers( pointers to int)
int &x;// declaring a reference variable
cout << &x; //"address-of" operator, Get the address of X

int size = 6;
//a dynamic arrayt can be resized:
int * list;
list = new int[size];
//1) Find space for a new array that's 5 bigger
int * temp = new int[size +5];
//2) copy old data to new location
for( int i  = 0; i <size ; i++)
    temp[i]=list[i];
//3) rename the new array to the original name 
list = temp;