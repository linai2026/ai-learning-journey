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
        seen = {} # Dictionary to store seennumbers and their indices
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return (seen[complement], i)
            seen[num] = i
        return None