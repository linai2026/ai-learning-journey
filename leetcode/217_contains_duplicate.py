class Solution:

    #hash set(多使用一点 memory，可以降低 time complexity)
    def containsDuplicate_set(self, nums: List[int]) -> bool:
        num_set = set()
        for num in nums:
            #harhset lookup is O(1) on average
            if num in num_set:
                return True
            num_set.add(num)
        return False
    
    
    #brute force
     def containsDuplicate_bf(self, nums: List[int]) -> bool:
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] == nums[j]:
                    return True
        return False
            

    #sorting
    def containsDuplicate_sorting(self, nums: List[int]) -> bool:
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                return True
        return False