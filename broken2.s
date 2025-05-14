#name : William Brandon
#course : CDA 3100
#Assignment: Calculate Sum, Min, Max, Mean, and Variance

#variable declarations
.data
    prompt_num:     .asciiz "Enter the number of values to be read: "
    error_msg:      .asciiz "Value must be between 1 and 10\n"
    prompt_values:  .asciiz "Enter "
    prompt_values2: .asciiz " integer values, one per line:\n"
    sum_msg:        .asciiz "Sum is: "
    min_msg:        .asciiz "Min is: "
    max_msg:        .asciiz "Max is: "
    mean_msg:       .asciiz "Mean is: "
    var_msg:        .asciiz "Variance is: "
    newline:        .asciiz "\n"

.text
.globl main

main:
    #Function prologue
    addi $sp, $sp, -4 #allocate stack space
    sw $ra, 0($sp) #saves the return address 
input_loop:
    #prompt for the numbers from user
    la $a0, prompt_num
    jal prtmsg

    #Read number of values
    li $v0, 5
    syscall
    move $s0, $v0 #stores the value in the $v0 register

    #check if the nummber is between 1-10
    blt $s0, 1, input_error
    bgt $s0, 10, input_error

    #allocate array 
    sll $t0, $s0, 2 #multiply by 4 for byte size
    sub $sp, $sp, $t0 #allocates space on stack
    move $s1, $sp #saves array address in $s1

    #prompt users for the entering values into the array
    la $a0, prompt_values
    move $a1, $s0
    jal prtintmsg
    la $a0, prompt_values2
    jal prtmsg

    #read values into the array
    move $a0, $s1 #Array address
    move $a1, $s0 #number of elements in the array
    jal readvals

    #calculate and print sum
    move $a0, $s1 
    move $a1, $s0
    jal calcsum
    move $s2, $v0 #Store sum in $s2

    #find and print minimum
    move $a0, $s1 
    move $a1, $s0
    jal findmin 

    #find and print maximum
    move $a0, $s1 
    move $a1, $s0
    jal findmax

    #calculate and print mean
    move $a0, $s2 #sum
    move $a1, $s0 #count
    jal calcmean
    mov.d $f12, $f0 #store mean in $f12 (double precision)

    #calculate and print variance
    move $a0, $s1 
    move $a1, $s0
    mov.d $f2, $f12 #the mean (double precision)
    jal calcvar 

    # Function epilogue
    lw $ra, 0($sp)       #restores return address
    addi $sp, $sp, 4     #deallocates stack space
    jr $ra               #return from main

input_error:
    la $a0, error_msg
    jal prtmsg
    j input_loop

#function implmentation
# /* print the msg string passed as an argument */
# void prtmsg(char *msg);
prtmsg:
    li $v0, 4
    syscall
    jr $ra

# /* print the msg string passed as the first argument using prtmsg,
#    followed by printing the integer val passed as the second argument,
#    followed by printing a newline character */
# void prtintmsg(char *msg, int val);
prtintmsg:
    #save arguments
    addi $sp, $sp, -8
    sw $a0, 0($sp)
    sw $a1, 4($sp)

    #print message
    jal prtmsg

    #print integer
    lw $a0, 4($sp)
    li $v0, 1
    syscall

    #print newline
    la $a0, newline
    li $v0, 4
    syscall

    #restore stack
    addi $sp, $sp, 8
    jr $ra

# /* print the msg string passed as the first argument using prtmsg,
#    followed by printing the double val passed as the second argument,
#    followed by printing a newline character */
# void prtdblmsg(char *msg, double val);
prtdblmsg:
    #save arguments
    addi $sp, $sp, -12
    sw $ra, 0($sp)
    sw $a0, 4($sp)
    s.d $f12, 8($sp)

    #print message
    jal prtmsg

    #print double
    l.d $f12, 8($sp)
    li $v0, 3
    syscall

    #print newline
    la $a0, newline
    li $v0, 4
    syscall

    #restore stack
    lw $ra, 0($sp)
    addi $sp, $sp, 12
    jr $ra

# /* read integer values into the array where the address of the array a is
#    passed as the first argument, and the number of values n is passed as
#    the second argument */
# void readvals(int a[], int n);
readvals:
    move $t0, $a0        #array address
    move $t1, $a1        #number of elements in array
    li $t2, 0            #counter
read_loop:
    beq $t2, $t1, read_done
    li $v0, 5            #read integer
    syscall
    sw $v0, 0($t0)       #store integer in array
    addi $t0, $t0, 4     #next array element
    addi $t2, $t2, 1     #increment counter
    j read_loop
read_done:
    jr $ra

# /* calculate the sum of the array elements where the address of the array a
#    is passed as the first argument and the number of values n is passed as
#    the second argument and print the sum with the appropriate message using
#    prtintmsg, and return the sum from the function */
# int calcsum(int a[], int n);
calcsum:
    move $t0, $a0        # Array address
    move $t1, $a1        # Number of elements
    li $t2, 0            # Counter
    li $v0, 0            # Initialize sum
sum_loop:
    beq $t2, $t1, sum_done
    lw $t3, 0($t0)       # Load array element
    add $v0, $v0, $t3    # Add to sum
    addi $t0, $t0, 4     # Next array element
    addi $t2, $t2, 1     # Increment counter
    j sum_loop
sum_done:
    # Print sum
    move $t4, $v0        # Save sum
    la $a0, sum_msg
    move $a1, $v0
    jal prtintmsg
    move $v0, $t4        # Restore sum
    jr $ra

# /* determine the minimum value of the array elements where the address of
#    the array a is passed as the first argument and the number of values n
#    is passed as the second argument and print the mininum value with the
#    appropriate message using prtintmsg */
# void findmin(int a[], int n);
findmin:
    move $t0, $a0        #array address
    move $t1, $a1        #number of elements
    lw $t2, 0($t0)       #initialize min with first element
    li $t3, 1            #counter (start from 1)
min_loop:
    beq $t3, $t1, min_done
    addi $t0, $t0, 4     #next array element
    lw $t4, 0($t0)       # Load array element
    bge $t4, $t2, min_next
    move $t2, $t4        #update min
min_next:
    addi $t3, $t3, 1     #increment counter
    j min_loop
min_done:
    #print min
    la $a0, min_msg
    move $a1, $t2
    jal prtintmsg
    jr $ra

# /* determine the maximum value of the array elements where the address of
#    the array a is passed as the first argument, and the number of values n
#    is passed as the second argument and print the maximum value with the
#    appropriate message using prtintmsg */
# void findmax(int a[], int n);
findmax:
    move $t0, $a0        #array address
    move $t1, $a1        #number of elements
    lw $t2, 0($t0)       #initialize max with first element
    li $t3, 1            #counter (start from 1)
max_loop:
    beq $t3, $t1, max_done
    addi $t0, $t0, 4     #move to next array element
    lw $t4, 0($t0)       #load array element
    ble $t4, $t2, max_next
    move $t2, $t4        #update max
max_next:
    addi $t3, $t3, 1     #increment counter
    j max_loop
max_done:
    # Print max
    la $a0, max_msg
    move $a1, $t2
    jal prtintmsg
    jr $ra

# /* calculate and return the mean of the array elements by using the sum
#    passed as the first argument and the number of values n passed as
#    the second argument and print the mean with the appropriate message
#    using prtdblmsg */
# double calcmean(int sum, int n);
calcmean:
    mtc1 $a0, $f0        #convert sum to double
    cvt.d.w $f0, $f0
    mtc1 $a1, $f2        #convert count to double
    cvt.d.w $f2, $f2
    div.d $f0, $f0, $f2  #calculate mean
    # Print mean
    la $a0, mean_msg
    mov.d $f12, $f0
    jal prtdblmsg
    jr $ra

# /* calculate the variance of the array elements where the address of the
#    array a is passed as the first argument, the number of values n is
#    passed as the second argument, and the mean is passed as the third
#    argument (double precision) and print the variance with the appropriate message using
#    prtdblmsg */
# void calcvar(int a[], int n, double mean);
calcvar:
    addi $sp, $sp, -16   #allocate stack space
    sw $ra, 0($sp)       #save return address
    sw $a0, 4($sp)       #save array address
    sw $a1, 8($sp)       #save count
    s.d $f2, 12($sp)     #save mean

    move $t0, $a0        #array address
    move $t1, $a1        #number of elements
    li $t2, 0            #counter
    mtc1 $zero, $f4
    mtc1 $zero, $f5      #initialize variance sum to 0 (double precision)

var_loop:
    beq $t2, $t1, var_done
    lw $t3, 0($t0)       #load array element
    mtc1 $t3, $f6        #convert to double
    cvt.d.w $f6, $f6
    sub.d $f8, $f6, $f2  #subtract mean
    mul.d $f8, $f8, $f8  #square the difference
    add.d $f4, $f4, $f8  #add to variance sum
    addi $t0, $t0, 4     #next array element
    addi $t2, $t2, 1     #increment counter
    j var_loop

var_done:
    lw $t1, 8($sp)       #restore count
    mtc1 $t1, $f6        #convert count to double
    cvt.d.w $f6, $f6
    div.d $f0, $f4, $f6  #divide by count for final variance

    # Print variance
    la $a0, var_msg
    mov.d $f12, $f0
    jal prtdblmsg

    lw $ra, 0($sp)       #restore return address
    addi $sp, $sp, 16    #deallocate stack space
    jr $ra