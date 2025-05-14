.data
    prompt_num:     .asciiz "Enter the number of values to be read\n"
    error_msg:      .asciiz "Value must be between 1 and 10\n"
    prompt_values:  .asciiz "Enter "
    prompt_values2: .asciiz " integer values, one per line\n"
    sum_msg:        .asciiz "Sum is: "
    min_msg:        .asciiz "Min is: "
    max_msg:        .asciiz "Max is: "
    mean_msg:       .asciiz "Mean is: "
    var_msg:        .asciiz "Variance is: "
    newline:        .asciiz "\n"
    array:          .word 0:10  # Array to store up to 10 integers
    debug_after_readvals:  .asciiz "\nDebug: After readvals\n"
    debug_after_calcsum:   .asciiz "\nDebug: After calcsum\n"
    debug_after_findmin:   .asciiz "\nDebug: After findmin\n"
    debug_after_findmax:   .asciiz "\nDebug: After findmax\n"
    debug_after_calcmean:  .asciiz "\nDebug: After calcmean\n"
    debug_after_calcvar:   .asciiz "\nDebug: After calcvar\n"

.text
.globl main

main:
    # Function prologue
    addi $sp, $sp, -4    # Allocate stack space
    sw $ra, 0($sp)       # Save the return address 

input_loop:
    # Prompt for the number of values
    la $a0, prompt_num
    jal prtmsg

    # Read number of values
    li $v0, 5
    syscall
    move $s0, $v0        # Store the number of values in $s0

    # Check if the number is between 1-10
    blt $s0, 1, input_error
    bgt $s0, 10, input_error

    # Prompt user for entering values into the array
    la $a0, prompt_values
    move $a1, $s0
    jal prtintmsg
    la $a0, prompt_values2
    jal prtmsg

    # Read values into the array
    la $a0, array        # Array address
    move $a1, $s0        # Number of elements
    jal readvals

    # Debug: Print after readvals
    #la $a0, debug_after_readvals
    #jal prtmsg

    # Calculate and print sum
    la $a0, array
    move $a1, $s0
    jal calcsum
    move $s1, $v0        # Store sum in $s1 (low 32 bits)
    move $s2, $v1        # Store sum in $s2 (high 32 bits)

    # Debug: Print after calcsum
    #la $a0, debug_after_calcsum
    #jal prtmsg

    # Find and print minimum
    la $a0, array
    move $a1, $s0
    jal findmin 

    # Debug: Print after findmin
    #la $a0, debug_after_findmin
    #jal prtmsg

    # Find and print maximum
    la $a0, array
    move $a1, $s0
    jal findmax
    move $a1, $v0        # Move result to $a1 for printing
    la $a0, max_msg      # Load address of max message
    jal prtintmsg        # Print the max value

    # Debug: Print after findmax
    #la $a0, debug_after_findmax
    #jal prtmsg

    # Calculate and print mean
    move $a0, $s1        # Low 32 bits of sum
    move $a1, $s2        # High 32 bits of sum
    move $a2, $s0        # Number of elements
    jal calcmean
    mov.s $f12, $f0      # Move result to $f12 for printing
    la $a0, mean_msg     # Load address of mean message
    jal prtfpmsg         # Print the mean value

    # Debug: Print after calcmean
    #la $a0, debug_after_calcmean
    #jal prtmsg

    # Calculate and print variance
    la $a0, array
    move $a1, $s0        # Number of elements
    mov.s $f12, $f0      # Mean (result from calcmean)
    jal calcvar 
    mov.s $f12, $f0      # Move result to $f12 for printing
    la $a0, var_msg      # Load address of variance message
    jal prtfpmsg         # Print the variance value

    # Debug: Print after calcvar
    #la $a0, debug_after_calcvar
    #jal prtmsg

    # Function epilogue
    lw $ra, 0($sp)       # Restore return address
    addi $sp, $sp, 4     # Deallocate stack space
    jr $ra               # Return from main

input_error:
    la $a0, error_msg
    jal prtmsg
    j input_loop

# Print a null-terminated string
# void prtmsg(char *msg);
prtmsg:
    li $v0, 4
    syscall
    jr $ra

# Print a string followed by an integer and a newline
# void prtintmsg(char *msg, int val);
prtintmsg:
    # Save arguments and return address
    addi $sp, $sp, -16   # Allocate 16 bytes (4 words) to ensure 16-byte alignment
    sw $ra, 0($sp)
    sw $a0, 4($sp)
    sw $a1, 8($sp)

    # Print message
    jal prtmsg

    # Print integer
    lw $a0, 8($sp)
    li $v0, 1
    syscall

    # Print newline
    la $a0, newline
    li $v0, 4
    syscall

    # Restore stack and return
    lw $ra, 0($sp)
    addi $sp, $sp, 16    # Deallocate 16 bytes
    jr $ra

# Print a string followed by a float and a newline
# void prtfpmsg(char *msg, float val);
prtfpmsg:
    # Save arguments and return address
    addi $sp, $sp, -12
    sw $ra, 0($sp)
    sw $a0, 4($sp)
    s.s $f12, 8($sp)

    # Print message
    jal prtmsg

    # Print float
    l.s $f12, 8($sp)
    li $v0, 2
    syscall

    # Print newline
    la $a0, newline
    li $v0, 4
    syscall

    # Restore stack and return
    lw $ra, 0($sp)
    addi $sp, $sp, 12
    jr $ra

# Read integer values into the array
# void readvals(int a[], int n);
readvals:
    move $t0, $a0        # Array address
    move $t1, $a1        # Number of elements
    li $t2, 0            # Counter
read_loop:
    beq $t2, $t1, read_done
    li $v0, 5            # Read integer
    syscall
    sw $v0, 0($t0)       # Store integer in array
    addi $t0, $t0, 4     # Next array element
    addi $t2, $t2, 1     # Increment counter
    j read_loop
read_done:
    jr $ra

# Calculate the sum of the array elements
# long long calcsum(int a[], int n);
calcsum:
    addi $sp, $sp, -4    # Allocate stack space
    sw $ra, 0($sp)       # Save return address

    move $t0, $a0        # Array address
    move $t1, $a1        # Number of elements
    li $t2, 0            # Counter
    li $v0, 0            # Sum (low 32 bits)
    li $v1, 0            # Sum (high 32 bits)
sum_loop:
    beq $t2, $t1, sum_done
    lw $t3, 0($t0)       # Load array element
    addu $v0, $v0, $t3   # Add to sum (low 32 bits) 
    sltu $t4, $v0, $t3   # Check for carry
    addu $v1, $v1, $t4   # Add carry to high 32 bits
    addi $t0, $t0, 4     # Next array element
    addi $t2, $t2, 1     # Increment counter
    j sum_loop
sum_done:
    # Save sum to temporary registers
    move $t5, $v0        # Save low 32 bits of sum
    move $t6, $v1        # Save high 32 bits of sum

    # Print sum
    la $a0, sum_msg
    move $a1, $t5        # Low 32 bits of sum
    jal prtintmsg

    # Restore sum to v0 and v1
    move $v0, $t5        # Restore low 32 bits of sum
    move $v1, $t6        # Restore high 32 bits of sum

    lw $ra, 0($sp)       # Restore return address
    addi $sp, $sp, 4     # Deallocate stack space
    jr $ra               # Return to caller

# Find the minimum value in the array
# void findmin(int a[], int n);
findmin:
    # Save return address
    addi $sp, $sp, -4
    sw $ra, 0($sp)

    move $t0, $a0        # Array address
    move $t1, $a1        # Number of elements
    lw $t2, 0($t0)       # Initialize min with first element
    li $t3, 1            # Counter (start from 1)
min_loop:
    beq $t3, $t1, min_done
    addi $t0, $t0, 4     # Next array element
    lw $t4, 0($t0)       # Load array element
    bge $t4, $t2, min_next
    move $t2, $t4        # Update min
min_next:
    addi $t3, $t3, 1     # Increment counter
    j min_loop
min_done:
    # Print min
    la $a0, min_msg
    move $a1, $t2
    jal prtintmsg

    # Restore return address and return
    lw $ra, 0($sp)
    addi $sp, $sp, 4
    jr $ra

# Find the maximum value in the array
# int findmax(int a[], int n);
findmax:
    move $t0, $a0        # Array address
    move $t1, $a1        # Number of elements
    lw $t2, 0($t0)       # Initialize max with first element
    li $t3, 1            # Counter (start from 1)
max_loop:
    beq $t3, $t1, max_done
    addi $t0, $t0, 4     # Move to next array element
    lw $t4, 0($t0)       # Load array element
    ble $t4, $t2, max_next
    move $t2, $t4        # Update max
max_next:
    addi $t3, $t3, 1     # Increment counter
    j max_loop
max_done:
    move $v0, $t2        # Return max value in $v0
    jr $ra

# Calculate the mean
# float calcmean(int sumLow, int sumHigh, int n);
calcmean:
    # Convert low 32 bits to double
    mtc1.d $a0, $f0
    cvt.d.w $f0, $f0
    
    # Convert high 32 bits to double and combine with low bits
    mtc1.d $a1, $f2
    cvt.d.w $f2, $f2
    lui $t0, 0x4330    # Load upper half of 2^32 (4503599627370496.0)
    mtc1.d $t0, $f4
    sub.d $f4, $f4, $f6   # $f6 should contain 2^32 as a double
    mul.d $f2, $f2, $f4   # Multiply high bits by 2^32
    add.d $f0, $f0, $f2   # Combine low and high bits
    
    # Convert count to double
    mtc1.d $a2, $f2
    cvt.d.w $f2, $f2
    
    # Calculate mean
    div.d $f0, $f0, $f2
    
    # Convert result to single precision
    cvt.s.d $f0, $f0
    
    jr $ra               # Return with result in $f0

# Calculate the variance
# float calcvar(int a[], int n, float mean);
calcvar:
    addi $sp, $sp, -4    # Allocate stack space
    sw $ra, 0($sp)       # Save return address

    move $t0, $a0        # Array address
    move $t1, $a1        # Number of elements
    mov.s $f4, $f12      # Mean (store in $f4 for safekeeping)
    cvt.d.s $f4, $f4     # Convert mean to double precision
    li $t2, 0            # Counter
    mtc1 $zero, $f6
    mtc1 $zero, $f7      # Initialize variance sum to 0 (double precision)

var_loop:
    beq $t2, $t1, var_done
    lw $t3, 0($t0)       # Load array element
    mtc1 $t3, $f8
    cvt.d.w $f8, $f8     # Convert to double
    sub.d $f10, $f8, $f4 # Subtract mean
    mul.d $f10, $f10, $f10 # Square the difference
    add.d $f6, $f6, $f10 # Add to variance sum
    addi $t0, $t0, 4     # Next array element
    addi $t2, $t2, 1     # Increment counter
    j var_loop

var_done:
    mtc1 $t1, $f8
    cvt.d.w $f8, $f8     # Convert count to double
    div.d $f0, $f6, $f8  # Divide by count for final variance
    cvt.s.d $f0, $f0     # Convert result to single precision

    lw $ra, 0($sp)       # Restore return address
    addi $sp, $sp, 4     # Deallocate stack space
    jr $ra               # Return with result in $f0s