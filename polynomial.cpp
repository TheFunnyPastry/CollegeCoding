#include "polynomial.h"
#include <cctype>
#include <cmath>

using namespace std;

//default constructor
Polynomial::Polynomial() : letter('x'), degree(-1) {
    clear();
}

//parameterized constructor
Polynomial::Polynomial(int constant) : letter('x') {
    clear();
    for(int i = 0; i <= MAX_DEGREE; i++) {
        coefficients[i] = constant - i;
    }
    updateDegree();
}

void Polynomial::clear() {
    for(int i = 0; i <= MAX_DEGREE; i++) {
        coefficients[i] = 0;
    }
    degree = -1;
}

int Polynomial::evaluate(int value) const {
    int result = 0;
    for(int i = 0; i <= degree; i++) {
        result += coefficients[i] * pow(value, i);
    }
    return result;
}

int Polynomial::getCoefficient(int k) const {
    if(k >= 0 && k <= degree) {
        return coefficients[k];
    }
    return 0;
}

int Polynomial::getDegree() const {
    return degree;
}

bool Polynomial::setCoefficient(int k, int value) {
    if(k >= 0 && k <= MAX_DEGREE) {
        coefficients[k] = value;
        updateDegree();
        return true;
    }
    return false;
}

bool Polynomial::setLetter(char newLetter) {
    if(isalpha(newLetter)) {
        letter = tolower(newLetter);
        return true;
    }
    return false;
}

void Polynomial::updateDegree() {
    degree = -1;
    for(int i = MAX_DEGREE; i >= 0; i--) {
        if(coefficients[i] != 0) {
            degree = i;
            break;
        }
    }
}

int Polynomial::sumCoefficients() const {
    int sum = 0;
    for(int i = 0; i <= degree; i++) {
        sum += coefficients[i];
    }
    return sum;
}

//operator overloading implementations
ostream& operator<<(std::ostream& out, const Polynomial& p) {
    if(p.degree == -1) {
        out << "Polynomial of degree -1";
        return out;
    }

    bool first = true;
    for(int i = p.degree; i >= 0; i--) {
        if(p.coefficients[i] != 0) {
            if(!first && p.coefficients[i] > 0) out << "+ ";
            if(p.coefficients[i] < 0) out << "- ";
            
            int coeff = abs(p.coefficients[i]);
            if((coeff != 1 || i == 0) && coeff != 0) out << coeff;
            
            if(i > 0) {
                out << p.letter;
                if(i > 1) out << "^" << i;
            }
            if(i > 0) out << " ";
            first = false;
        }
    }
    return out;
}

istream& operator>>(std::istream& in, Polynomial& p) {
    char letter;
    char semicolon;
    in >> letter >> semicolon;

    if(!isalpha(letter)) {
        p.clear();
        return in;
    }

    p.setLetter(letter);
    
    for(int i = 0; i <= Polynomial::MAX_DEGREE; i++) {
        int coeff;
        in >> coeff;
        p.coefficients[i] = coeff;
        if(i < Polynomial::MAX_DEGREE) {
            char comma;
            in >> comma;
        }
    }
    p.updateDegree();
    return in;
}

//arithmetic operators
Polynomial Polynomial::operator+(const Polynomial& other) const {
    Polynomial result;
    result.letter = this->letter;
    
    for(int i = 0; i <= MAX_DEGREE; i++) {
        result.coefficients[i] = this->coefficients[i] + other.coefficients[i];
    }
    result.updateDegree();
    return result;
}

Polynomial Polynomial::operator-(const Polynomial& other) const {
    Polynomial result;
    result.letter = this->letter;
    
    for(int i = 0; i <= MAX_DEGREE; i++) {
        result.coefficients[i] = this->coefficients[i] - other.coefficients[i];
    }
    result.updateDegree();
    return result;
}

Polynomial Polynomial::operator*(const Polynomial& other) const {
    Polynomial result;
    result.letter = this->letter;
    
    for(int i = 0; i <= degree; i++) {
        for(int j = 0; j <= other.degree; j++) {
            if(i + j <= MAX_DEGREE) {
                result.coefficients[i + j] += coefficients[i] * other.coefficients[j];
            }
        }
    }
    result.updateDegree();
    return result;
}

//comparison operators
bool Polynomial::operator==(const Polynomial& other) const {
    if(degree != other.degree) return false;
    for(int i = 0; i <= degree; i++) {
        if(coefficients[i] != other.coefficients[i]) return false;
    }
    return true;
}

bool Polynomial::operator!=(const Polynomial& other) const {
    return !(*this == other);
}

bool Polynomial::operator<(const Polynomial& other) const {
    if(degree != other.degree) return degree < other.degree;
    return sumCoefficients() < other.sumCoefficients();
}

bool Polynomial::operator>(const Polynomial& other) const {
    return other < *this;
}

bool Polynomial::operator<=(const Polynomial& other) const {
    return !(other < *this);
}

bool Polynomial::operator>=(const Polynomial& other) const {
    return !(*this < other);
}