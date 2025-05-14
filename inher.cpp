#include <iostream>


int main(){
int weight = 0, snickers = 0, crunch = 0, kitkat = 0, reeses = 0;
std::cout<<"What is the weight of your candy ";
std::cin>> weight;

while(weight>0){
    while(weight-9>=0){
        weight-=9;
        reeses++;
    }
    while(weight-6>=0){
        weight-=6;
        kitkat++;
    }
    while(weight-3>=0){
        weight-=3;
        crunch++;
    }
    while(snickers-1>=0){
        weight-=1;
        snickers++;
    }
}
std::cout<<reeses<<" "<<kitkat<<" "<<crunch<<" "<<snickers;
std::cout<<weight;



    return 0;
}