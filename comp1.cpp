#include <iostream>

int main(){
int weight, maxVAL;
int candy[4]={9,6,3,1};

std:: cin>> weight;
for(int i = 0; i < 4; i++){
maxVAL = weight / candy[i];
if (maxVAL > 10) maxVAL = 10;
std:: cout<<maxVAL<<" ";
weight = weight-(maxVAL * candy[i]);
}
    return 0;
}