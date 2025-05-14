#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <string>
#include "sphere.h"

const double PI = 3.14159;

//helper function implementations
bool Sphere::validColor(char c) const {
    char upperC = toupper(c);
    return upperC == 'B' || upperC == 'R' || upperC == 'P' || 
           upperC == 'Y' || upperC == 'G' || upperC == 'L' || upperC == 'M';
}

bool Sphere::randColor(){
    const char colors[] = {'B', 'R', 'P', 'Y', 'G', 'L', 'M'};
    color = colors[rand() % 7];
    return true;
}

std::string Sphere::colortostr() const {
    switch(toupper(color)) {
        case 'B': return "Blue";
        case 'R': return "Red";
        case 'P': return "Purple";
        case 'Y': return "Yellow";
        case 'G': return "Green";
        case 'L': return "Black";
        case 'M': return "Maroon";
        default: return "Unknown";
    }
}

//constructors
//default
Sphere::Sphere() {
    radius = 1.0;
    randColor();
}
//two parameter constroctor
Sphere::Sphere(double r, char c) {
    if (r < 0.0){
        r = 1.0;}
    else{
        radius = r;
    }
    if (validColor(c)) {
        color = toupper(c);
    } else {
        randColor();
    }
}

// Getter functions
double Sphere::getRad() const {
    return radius;
}

char Sphere::getColor() const {
    return color;
}

double Sphere::getDiameter() const {
    return 2 * radius;
}

double Sphere::getSurfaceArea() const {
    return 4 * PI * radius * radius;
}

double Sphere::getVolume() const {
    return (4.0/3.0) * PI * radius * radius * radius;
}

// Setter functions
void Sphere::SetRadius(double r) {
    if (r > 0.0) {
        radius = r;
    }
}

void Sphere::SetColor(double r) {
    if (validColor(r)) {
        color = toupper(r);
    }
}

void Sphere::grow(double amount) {
    if (radius + amount > 0.0) {
        radius += amount;
    }
}

void Sphere::shrink(double amount) {
    if (radius - amount > 0.0) {
        radius -= amount;
    }
}

void Sphere::randomizeColor() {
    randColor();
}

void Sphere::printSummary(int precision) {
    if (precision < 1 || precision > 5) {
        precision = 2;
    }
    
    std::cout << "Radius: " << std::fixed << std::setprecision(precision) 
              << radius << std::endl;
    std::cout << "Color: " << colortostr() << std::endl;
    std::cout << "Diameter: " << std::fixed << std::setprecision(precision) 
              << getDiameter() << std::endl;
    std::cout << "Volume: " << std::fixed << std::setprecision(precision) 
              << getVolume() << std::endl;
    std::cout << "Surface Area: " << std::fixed << std::setprecision(precision) 
              << getSurfaceArea() << std::endl;
}