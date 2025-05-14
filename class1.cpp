#include <iostream>

using namespace std;

int main(){
int num1,num2,sum;


cout<< "Put in your two numbers for the interval: ";
cin>>num1;

cin>>num2;

for(int i = num1; i< num2; i++){
if(i % 2 ==1){
sum += i;
}
}
cout<< sum;


return 0;
}
