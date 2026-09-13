class Solution:

    # time complexity: O(n) ; space complexity: O(1)
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1
        
        while left < right:
            current_sum = numbers[left] + numbers[right]
            
            if current_sum == target:
                return [left + 1, right + 1]  # Return 1-based indices

            # Move the pointers based on the comparison of current_sum and target
            elif current_sum < target:
                left += 1
            else:
                right -= 1
        
        return []  # Return an empty list if no solution is found
