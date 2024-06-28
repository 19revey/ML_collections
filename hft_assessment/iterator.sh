#!/bin/bash

# Define the array
my_array=("one" "two" "three")

# Initialize the counter
counter=0

# Define a variable to store the value
value=""

# Define the function to get the next element
getnext() {
    # Get the current element from the array
    current_element="${my_array[$counter]}"
    
    # Increment the counter
    counter=$(( (counter + 1) % ${#my_array[@]} ))
    
    # Set the value variable to the current element
    value="$current_element"
}

# Iterate over the array
for i in $(seq 1 5); do
    # Call the function to get the next element
    getnext
    
    # Print the value of the current element
    echo "Value: $value"
done
