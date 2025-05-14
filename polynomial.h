#define POLYNOMIAL_H

#include <iostream>

class Polynomial {

public:
    //constructors
    Polynomial();                      //default constructor
    Polynomial(int constant);          //parameterized constructor

    //member functions
    void clear();
    int evaluate(int value = 0) const;
    int getCoefficient(int k) const;
    int getDegree() const;
    bool setCoefficient(int k, int value = 0);
    bool setLetter(char newLetter);

    //friend operators
    friend ostream& operator<<(ostream& out, const Polynomial& p);
    friend istream& operator>>(istream& in, Polynomial& p);

    //arithmetic operators
    Polynomial operator+(const Polynomial& other) const;
    Polynomial operator-(const Polynomial& other) const;
    Polynomial operator*(const Polynomial& other) const;

    //comparison operators
    bool operator==(const Polynomial& other) const;
    bool operator!=(const Polynomial& other) const;
    bool operator<(const Polynomial& other) const;
    bool operator>(const Polynomial& other) const;
    bool operator<=(const Polynomial& other) const;
    bool operator>=(const Polynomial& other) const;

private:
    static const int MAX_DEGREE = 10;  //can be modified through by reference operation
    int coefficients[MAX_DEGREE + 1];  //array to store coefficients
    char letter;                       //variable letter (x, y, etc.)
    int degree;                        //current degree of polynomial

    void updateDegree();  // helper function to update polynomial degree
    int sumCoefficients() const;  //helper function for comparison operators
};