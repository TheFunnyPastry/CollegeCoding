#include <iostream> 
#include <cctype>

int C2I(char c)
//Converts character into integer (returns -1 for error)
{
   if (c < '0' || c > '9')
      return -1;     //error
   return (c - '0'); //success
}

int main(){
int size = 12;
char c;
char arr[size]={};
int t = 0;

for(int i = 0;i<size;i++){
    std::cin>>arr[i];
    t++;
    if(arr[i]=='\n'){
        for(int i = 0;i<size-t;i++){
            arr[i] = 0;
        }
        break;
    }
}

for(int i = 0;i<size;i++){
    std::cout<<arr[i];
}

int g = 0;
int counter = 0;
while(isdigit(arr[g])||arr[g]==','){
    if(isdigit(arr[g])){
        counter++;
        g++;
    }
    else if(arr[g]==','){
        g++;
    }
}

std::cout<<counter;
int i = 0;

int TECH[size]={};

/*for(i = 0; i < counter; i++){
    if(isdigit(arr[i])&&arr[i]!=','){
    if (arr[i] < '0' || arr[i] > '9')
      return -1;     //error
   TECH[i]=(arr[i] - '0'); //success
    }
}*/

for(i = 0; i < size; i+=2){
    TECH[i]=C2I(arr[i]);          //instead just add 2 to i 
}


//for loop for tech
for(i = 0; i < size; i+=2){
    std::cout<<TECH[i];
}
int j = 0;
/*while(TECH[0]+TECH[2]==TECH[4]&&j<size){
    std:: cout<<"double check";
    if(TECH[j]+TECH[j+2]==TECH[j+4]){
        j+=2;
        std::cout<<"is working";
    }
}*/

if(TECH[0]+TECH[2]==TECH[4]){
    for(int i = 0; i < size; i+=2){
        if(TECH[i]+TECH[i+2]==TECH[i+4]){
            return true;
        }
        else
            return false;
    }
}


return 0;

}
