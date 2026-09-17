class Solution:
    # time O(n), extra space O(1) 
    def maxProfit(self, prices: List[int]) -> int:
        
        min_price = float('inf')
        max_profit = 0
        
        for price in prices:
            if price < min_price:
                min_price = price
            elif price - min_price > max_profit:
                max_profit = price - min_price  

        return max_profit








