# Brute Force O(n^2) solution
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in nums and nums.index(complement) != i:
               return [i, nums.index(complement)]
                
            
print(Solution().twoSum([2, 7, 11, 15], 9))
print(Solution().twoSum([3,2,4], 6))
print(Solution().twoSum([3,3], 6))

# Hash Map O(n) solution
class Solution_Hash:
    def twoSum(self, nums: list[int], target: int) -> list[int]:

        # Dictionary to store seen numbers and their indices
        seen = {} 
        
        for i, num in enumerate(nums):
            # Calculate the complement of the current number
            complement = target - num

            # Check if the complement exists in the seen dictionary
            if complement in seen:
                # If the complement is found, return the indices of the two numbers
                return (seen[complement], i)
            # Store the current number and its index in the seen dictionary
            seen[num] = i
        return None