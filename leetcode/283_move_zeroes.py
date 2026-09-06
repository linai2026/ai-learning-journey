class Solution:
    def moveZeroes(self, nums: List[int]) -> None
        """
        Do not return anything, modify nums in-place instead.
        """
        # Initialize a pointer for the position of the next non-zero element
        non_zero_index = 0
        
        # Move all non-zero elements to the front of the array
        for num in nums:
            if num != 0:
                nums[non_zero_index] = num
                non_zero_index += 1
        
        # Fill the remaining positions with zeros
        for i in range(non_zero_index, len(nums)):
            nums[i] = 0