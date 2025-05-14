.data
    veclen: .asciiz "Enter the number of elements in each vector: "
    numlow: .asciiz "Error: number must be at least 1\n"
    numhigh: .asciiz "Error: number must be lower than 10\n"
    vector1: .asciiz "Enter the elements of the first vector:\n"
    vector2: .asciiz "Enter the elements of the second vector:\n"
    sum: .asciiz "The vector sum is:\n"
    product: .asciiz "The vector product is:\n"
    vec1: .word 0:10  # Reserve space for up to 10 integers for first vector
    vec2: .word 0:10  # Reserve space for up to 10 integers for second vector

.text
.globl main
main:
    #prints out prompt for vector length
    li $v0, 4
    la $a0, veclen
    syscall

    # Read integer
    li $v0, 5
    syscall
    move $s0, $v0  #stores size in $s0

    # Check if size more than 1
    li $t0, 1
    blt $s0, $t0, size_too_low

    # Check if size less than 10
    li $t0, 10
    bge $s0, $t0, size_too_high

    #if value more than 1 less than ten continue_program
    j continue_program

size_too_low:
    li $v0, 4
    la $a0, numlow
    syscall
    j exit_program

size_too_high:
    li $v0, 4
    la $a0, numhigh
    syscall
    j exit_program

continue_program:
    #reads in elements for vec1
    li $v0, 4
    la $a0, vector1
    syscall

    la $t0, vec1  # loads memory address of vec1
    li $t1, 0     # initializes counter

read_vec1_loop:
    beq $t1, $s0, read_vec2  # once the counter reaches the max value of veclen then move onto next vector
    
    li $v0, 5     #reads integer
    syscall
    sw $v0, ($t0) #stores integer in vector
    
    addi $t0, $t0, 4  #next element in the vector
    addi $t1, $t1, 1  #increment counter
    j read_vec1_loop

read_vec2:
    #reads in elements for vec2
    li $v0, 4
    la $a0, vector2
    syscall

    la $t0, vec2  #loads address of vec2
    li $t1, 0     #resets counter to 0

read_vec2_loop:
    beq $t1, $s0, calculate_results  #once the vectors are full move to performing calculations
    li $v0, 5     # Read integer
    syscall
    sw $v0, ($t0) # Store integer in vector
    
    addi $t0, $t0, 4  # Move to next element in vector
    addi $t1, $t1, 1  # Increment counter
    j read_vec2_loop

calculate_results:
    #prints sum message
    li $v0, 4
    la $a0, sum
    syscall

    la $t0, vec1
    la $t1, vec2
    li $t2, 0  # Counter

sum_loop:
    beq $t2, $s0, print_product
    lw $t3, ($t0)
    lw $t4, ($t1)
    add $t5, $t3, $t4
    
    #prints sum of corresponnding elements
    move $a0, $t5
    li $v0, 1
    syscall
    
    #prints space
    li $v0, 11       # syscall for print character
    li $a0, 32       # ASCII code for space
    syscall
    
    addi $t0, $t0, 4
    addi $t1, $t1, 4
    addi $t2, $t2, 1
    j sum_loop

print_product:
    #newline print to  seperate between lines
    li $v0, 11       # syscall for print character
    li $a0, 10       # ASCII code for newline
    syscall

    #prints out the message of products
    li $v0, 4
    la $a0, product
    syscall

    la $t0, vec1
    la $t1, vec2
    li $t2, 0  #resets the counter for lenght

product_loop: #loop for printing out the products of the elements of the  vectors
    beq $t2, $s0, exit_program
    lw $t3, ($t0)
    lw $t4, ($t1)
    mul $t5, $t3, $t4
    
    #print product of the elements in both vector
    move $a0, $t5
    li $v0, 1
    syscall
    
    #seperates the products
    li $v0, 11       # syscall for print character
    li $a0, 32       # ASCII code for space
    syscall
    
    addi $t0, $t0, 4
    addi $t1, $t1, 4
    addi $t2, $t2, 1
    j product_loop

exit_program:
    li $v0, 10
    syscall