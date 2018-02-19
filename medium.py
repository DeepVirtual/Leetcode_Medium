# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 00:17:55 2017

@author: cz
"""

#2. Add Two Numbers
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        
        
        
        c1=l1
        c2=l2
        l=lcur=ListNode(0)
        carry=0
        
        while c1 and c2:
            
            digit=(c1.val+c2.val+carry)%10
            carry=(c1.val+c2.val+carry)//10
            lcur.val=digit
            
            if c1.next or c2.next or carry:
                lcur.next=ListNode(0)
                lcur=lcur.next
            
            c1=c1.next
            c2=c2.next
            
        while c1:
            
            digit=(c1.val+carry)%10
            carry=(c1.val+carry)//10
            lcur.val=digit
            
            if c1.next  or carry:
                lcur.next=ListNode(0)
                lcur=lcur.next
            c1=c1.next
            
        while c2:
            
            digit=(c2.val+carry)%10
            carry=(c2.val+carry)//10
            lcur.val=digit
            
            if c2.next  or carry:
                lcur.next=ListNode(0)
                lcur=lcur.next
            c2=c2.next
        
        print(l)
        print(lcur.val)
        if  carry:
           lcur.val=carry
        
        return l
           
#3. Longest Substring Without Repeating Characters          
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        start=0
        maxL=0
        d={}
        
        
        for i in range(len(s)):
            if s[i] in d and start<=d[s[i]]:
                start=d[s[i]]+1
            else:
              maxL=  max(maxL,i-start+1)
            d[s[i]]=i
        return maxL
                
#5. Longest Palindromic Substring
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        
        """
        if len(s)<2:
            return s
        self.maxL=0
        self.low=0
        
        def extend(s,k,l):
            while k>=0 and l<len(s) and s[k]==s[l] :
                k-=1
                l+=1
                if self.maxL <l-k-1:
                    
                   self.low=k+1
                   self.maxL=l-k-1
                   
        for i in range(len(s)-1):
            extend(s,i,i)
            extend(s,i,i+1)
        return s[self.low:self.low+self.maxL]
            
#6. ZigZag Conversion        
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """        
#n=numRows
#Δ=2n-2    1                           2n-1                         4n-3
#Δ=        2                     2n-2  2n                    4n-4   4n-2
#Δ=        3               2n-3        2n+1              4n-5       .
#Δ=        .           .               .               .            .
#Δ=        .       n+2                 .           3n               .
#Δ=        n-1 n+1                     3n-3    3n-1                 5n-5
#Δ=2n-2    n                           3n-2                         5n-4
          
        if numRows==1 or numRows>=len(s):
            return s
        
        index=0
        step=1
        res=['']*numRows
        for x in s:
            res[index]+=x
            if index==0:
                step=1
            elif index== numRows-1:
                step=-1
            index+=step
        return ''.join(res)
            


#8. String to Integer (atoi)
class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        
        import re
        if not str:
            return 0
        
        str=str.strip()
        str=re.findall('^[+\-]?\d+',str)
        
        try:
            str=int(''.join(str))
            num=int(str)
            maxN=2147483647
            minN=-2147483648
            if num>maxN:
                return maxN
            if num<minN:
                return minN
            return num
        except:
           return 0
            
#11. Container With Most Water            
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        maxarea=0
        l=0
        r=len(height)-1
        while l<r:
             maxarea=max(maxarea,min(height[l],height[r])*(r-l))
             if height[l]<height[r]:
                 l+=1
              
             else:
                  r+=1
        return maxarea
            
            
#12. Integer to Roman 
class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        
        strs = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        nums = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]  
        
        ret=''

        for i,j in enumerate(nums):
            while num >=j:
                ret+=strs[i]
                num-=j
            if num==0:
               return ret
                
#15. 3Sum       
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
                
        if not nums:
            return []
        nums.sort()
        
        ans=[]
        for i in range(len(nums)-2):
            if i>0 and nums[i]==nums[i-1]:
                continue
            l=i+1
            r=len(nums)-1
            while l<r:
                s=nums[i]+nums[l]+nums[r]
                if s>0:
                    r-=1
                elif s<0:
                    l+=1
                else:
                    ans.append(nums[i]+nums[l]+nums[r])
                    while l<r and nums[l]==nums[l+1]:
                         
                            l+=1
                    while l<r and nums[r]==nums[r-1]:
                             r-=1
                    
                    l+=1
                    r-=1
        return ans
                    
#16. 3Sum Closest
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if not nums:
            return 0
        
        
        nums.sort()
        
        result=nums[0]+nums[1]+nums[2]
        
        for i in range(len(nums)-2):
            l=i+1
            r=len(nums)-1
            
            while l<r:
                sumN=nums[i]+nums[l]+nums[r]
                if sumN==target:
                    return target
                if abs(sumN-target)<abs(result-target):
                    result=sumN
                    
                if sumN<target:
                    l+=1
                elif sumN>target:
                    r-=1
                    
                    
        return result
                    
#17. Letter Combinations of a Phone Number                    
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """        
        
        numletter = {
                    '2': 'abc',
                    '3': 'def',
                    '4': 'ghi',
                    '5': 'jkl',
                    '6': 'mno',
                    '7': 'pqrs',
                    '8': 'tuv',
                    '9': 'wxyz'
                   }
        
        if not digits:
           return []
     
        if len(digits)==1:
            return list(numletter[digits])
        
        
        
        letters=[numletter[i] for i in digits]
        
        from functools import reduce
        
        return reduce((lambda x,y: [ i+j for i in x for j in y ]),letters )
        
#18. 4Sum        
class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def findNsum(nums,target,N,result,results):
            if len(nums)<N or N<2 or N*nums[0]>target or N*nums[-1]<target:
                return 
            
            if N==2:
               l=0
               r=len(nums)-1
               while l<r:
                   s=nums[l]+nums[r]
                   if s==target:
                      results.append(result+[nums[l],nums[r]])
                      l+=1
                      r-=1
                      while l<r and nums[l]==nums[l-1]:
                          l+=1
                      while l<r and nums[r]==nums[r+1]:
                          r-=1  
                   elif s>target:
                        r-=1
                        
                   else:
                        l+=1
            else:
                for i in range(len(nums)-N+1):
                    if i==0 or (i>0 and nums[i]!=nums[i-1]):
                       findNsum(nums[i+1:],target-nums[i],N-1,result+[nums[i]],results)
        results=[]
        findNsum(sorted(nums), target,4,[],results)
        return results
        
#19. Remove Nth Node From End of List        
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        if not head:
            return None
        
        dummy= ListNode(0)
        dummy.next=head
        
        
        first=second=dummy
        
        while n+1:
             second=second.next
             n=n-1
             
        
        while second:
            first=first.next
            second=second.next
            
        first.next=first.next.next
        return dummy.next
        
#22. Generate Parentheses        
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        def dfs(p,left,right,n):
            if right==n:
                self.ans.append(p)
                return 
            if left:
               dfs(p+'(',left-1,right,n)
            if right>left:
               dfs(p+')',left,right-1,n)
                   
        self.ans=[]
        dfs('',n,n,n)
        return self.ans
                
#24. Swap Nodes in Pairs        
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None
        
        if not head.next:
            return head
        
        
        #pre -> a -> b -> b.next to pre -> b -> a -> b.next
        dummy=ListNode(0)
        dummy.next=head
        
        pre,pre.next=dummy,head
        while pre.next  and pre.next.next:
            a=pre.next
            b=pre.next.next
            pre.next,b.next,a.next=b,a,b.next
            
            pre=a
        return dummy.next
            
        NH=head.next
        head.next=self.swapPairs(NH.next)
        NH.next=head
        return NH
        
#29. Divide Two Integers        
class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        if (dividend<0 and divisor >0)  or (dividend>0 and divisor <0):
           isPositive = False
        else:
           isPositive = True
          
        dividend=abs(dividend)
        divisor=abs(divisor)
        res=0
        c=1
        sub=divisor
        
        while  dividend >=divisor:
            
            if dividend >= sub:
                
               dividend=dividend-sub
               res+=c
               sub=(sub<<1)
               c=(c<<1)
            else:
                 sub=(sub>>1)
                 c=(c>>1)
        if not isPositive:
            res=-res
        
        return min(max(-2147483648,res),2147483647)
        
#31. Next Permutation
class Solution(object):
    
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if  len(nums) < 2:
           return 
       
        i=len(nums)-1
        while i>0 and nums[i]<=nums[i-1] :
              i-=1
        index1=i-1
        if index1<0:
            a=index1+1
            b=len(nums)-1
        
            while a<b:
               nums[a], nums[b]= nums[b], nums[a] 
               a+=1
               b-=1
            print(nums)
            return 
        k=i
        print(i,k)
        while  k< len(nums) and nums[index1] < nums[k] :
                           
              k+=1
        
        
        index2=k-1
        
        print(index1,index2)
        nums[index1],nums[index2] = nums[index2] , nums[index1]
#        print(index1)
#        print(nums[index1+1:])
             
        print(nums)
        a=index1+1
        b=len(nums)-1
        
        while a<b:
           nums[a], nums[b]= nums[b], nums[a] 
           a+=1
           b-=1
            
        print(nums)
        
if __name__ == "__main__":
    print(Solution().nextPermutation([1,5,1])) 
              
        
#33. Search in Rotated Sorted Array        
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """        
       
                
        l=0
        r=len(nums)-1
       
       
       
        while (l<=r):
           mid=l+(r-l)//2
           if nums[mid]==target:
               return mid
           if nums[mid] < nums[r]:# right half sorted
               if nums[mid] < target and target <= nums[r]:
                   l=mid+1
               else:
                   r=mid-1
           else:# left half sorted
               if nums[mid] > target and target >= nums[l]:
                   r=mid-1
               else:
                   l=mid+1
        return -1
               
if __name__ == "__main__":
    print(Solution().nextPermutation([1,5,1])) 
                               
#34. Search for a Range               
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if not nums:
            return [-1,-1]
        
        l=0
        r=len(nums)-1
        
        #get the left most index
        while l<=r:
            mid=(l+r)//2
            
            if nums[mid]<target:
                l=mid+1
            else:
                r=mid-1
            print(l,r)
        leftmost=l
        #print(leftmost)
        # if target not in list , index could return the left  insertion point. could be out of range
        if l <0 or l> (len(nums)-1) or  nums[l]!=target:
            return [-1,-1]
        
        #get the right most index
        
        l=0
        r=len(nums)-1
        while l<=r:
            mid=(l+r)//2
            
            if nums[mid]>target:
                 r=mid-1
            else:
                 l=mid+1
        rightmost=r        
        print(rightmost)
        return  [leftmost,rightmost]
        
if __name__ == "__main__":
    print(Solution().searchRange([2,2],3))         
        
            
#36. Valid Sudoku        
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        if not board:
            return True
        sumB=[]
        for i,row in enumerate(board):
            for j ,c in enumerate(row):
                if c!='.':
                    sumB+=[(i,c),(c,j),(i//3,j//3,c)]
        return len(sumB)==len(set(sumB))
        
#39. Combination Sum        
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        if not candidates:
            return []
        
        res=[]
        
        candidates.sort()
        def backtrack(res,path,candidates,remain,start):
            #print(self.tmp,self.res)
            if remain<0 :
                return 
            if remain==0:
                 res.append(path)
                 return 
         
            for i in range(start,len(candidates)):
                    
                    backtrack(res,path+[candidates[i]],candidates,remain-candidates[i],i)
                   
        backtrack(res,[],candidates,target,0)
        return res
                    
if __name__ == "__main__":
    print(Solution().combinationSum([2, 3, 6, 7],7))                     
                
               
#40. Combination Sum II
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
                
        if not candidates:
            return []
        
        res=[]
        
        candidates.sort()
        def backtrack(res,path,candidates,remain,start):
            #print(self.tmp,self.res)
            if remain<0 :
                return 
            if remain==0:
                 res.append(path)
                 return 
         
            for i in range(start,len(candidates)):
                if i>start and candidates[i]==candidates[i-1]:
                    continue
                    
                backtrack(res,path+[candidates[i]],candidates,remain-candidates[i],i+1)
                   
        backtrack(res,[],candidates,target,0)
        return res
                    
if __name__ == "__main__":
    print(Solution().combinationSum2([10, 1, 2, 7, 6, 1, 5],8))        
                   
 

#43. Multiply Strings 
class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        if not num1 or not num2:
            return 0
        if num1=='0'  or num2=='0':
            return '0'
        
        
        ret=[0]*[len(num1)+len(num2)]
        
        for i in range(len(num1)):
            for j in range(len(num2)):
                tmp=int(num1[i])*int(num2[j])
                ret(i+j)+=tmp//10
                ret(i+j=1)+=tmp%10
        carry=0
        
        for i in range(len(ret)-1,-1,-1):
        
           tmp2=ret[i]+carry
           ret[i]=tmp2%10
           carry=tmp2//10
           
        return ''.join(ret) if  ret[0] else ''.join(ret[1:])
        
#46. Permutations        
           
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res=[]
        
        def  dfs(nums,res,path):
            if not nums:
               res.append(path)
               return 
            for i in range(len(nums)):
                    tmp=nums[i]
                    
                    dfs(nums[:i]+nums[i+1:],res,path+[tmp])
        
        dfs(nums,res,[])
        return res
if __name__ == "__main__":
    print(Solution().permute([1,2,3]))        
                           
#47. Permutations II                
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
                        
        
        res=[]
        nums.sort()
        def  dfs(nums,res,path):
            if not nums:
               res.append(path)
               return 
            for i in range(len(nums)):
                    if i>0 and nums[i-1]==nums[i]:
                        continue
                    tmp=nums[i]
                    
                    dfs(nums[:i]+nums[i+1:],res,path+[tmp])
        
        dfs(nums,res,[])
        return res
if __name__ == "__main__":
    print(Solution().permuteUnique([1,1,3]))         
        
#48. Rotate Image
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
#/*
# * clockwise rotate
# * first reverse up to down, then swap the symmetry 
# * 1 2 3     7 8 9     7 4 1
# * 4 5 6  => 4 5 6  => 8 5 2
# * 7 8 9     1 2 3     9 6 3
#*/
#
#}
#
#/*
# * anticlockwise rotate
# * first reverse left to right, then swap the symmetry
# * 1 2 3     3 2 1     3 6 9
# * 4 5 6  => 6 5 4  => 2 5 8
# * 7 8 9     9 8 7     1 4 7
#*/
        
        
        matrix[::]=zip(*matrix[::-1])
        
        A[:] = [[row[i] for row in A[::-1]] for i in range(len(A))]

matrix=[[1, 2 ,3], [4, 5 ,6], [7, 8, 9]]



#list(zip(*matrix[::-1]))
if __name__ == "__main__":
    print(Solution().rotate([1,1,3])) 

A=matrix

for i in range(len(A)) :
    for row in A[::-1]:
      print(row[i],end=",")

#49. Group Anagrams
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
#        from collections import defaultdict 
#        ans=defaultdict(list)
#        
#        
#        
#        for s in strs:
#            count=[0]*26
#            for c in s:
#                count[ord(c)-ord('a')]+=1
#            
#            ans[tuple(count)].append(s) 
#        return ans.values()
    
    
        def groupAnagrams(self, strs):
            dic = {}
            for item in sorted(strs):
                sortedItem = ''.join(sorted(item))
                dic[sortedItem] = dic.get(sortedItem, []) + [item]
            return dic.values()
             
             
if __name__ == "__main__":
    print(Solution().groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])) 


#50. Pow(x, n)             
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n==0:
            return 1
        if n<0:
            n=-n
            x=1/x
        if n%2==0:
            return self.myPow(x*x,n//2)
        else:
            return x*self.myPow(x*x,n//2)


 

#54. Spiral Matrix
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        
        if not matrix:
            return []
        seen=[[False]*len(matrix[0]) for _ in matrix]
        
        dr=[0,1,0,-1]
        dc=[1,0,-1,0]
        
        r=0
        c=0
        di=0
        ans=[]
        for _ in range(len(matrix[0])*len(matrix)):
            print(r,c)
            print('***')
            ans.append(matrix[r][c])
            seen[r][c]=True
            cr=r+dr[di]
            cc=c+dc[di]
            print(cr,cc)
            
            if 0<=cr<len(matrix) and 0<=cc<len(matrix[0]) and not seen[cr][cc]:
                r=cr
                c=cc
                
            else:
                di=(di+1)%4
                r=r+dr[di]
                c=c+dc[di]
        return ans
        
if __name__ == "__main__":
    print(Solution().spiralOrder([[ 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8, 9 ]]))         

        
        
#55. Jump Game        
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """        
#A = [2,3,1,1,4], return true.
#A = [3,2,1,0,4], return false.  
        
        
        lastPos=len(nums)-1
        for i in range(len(nums)-2,-1,-1):
            if i+nums[i]>=lastPos:
               lastPos=i
        return lastPos==0
    
    
#56. Merge Intervals    
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        if not intervals:
            return []
        keep=0
        drop=[]
        for i in range(1,len(intervals)):
            if intervals[keep].e>=intervals[i].s:
                keep.e=intervals[i].e
                drop.append(i)
            else:
                keep=i
        
        return [intervals[x] for x  in range(len(intervals)) if x not in drop]     
        
if __name__ == "__main__":
    print(Solution().merge([[1,3],[2,6],[8,10],[15,18]]))           
        
        
        
#59. Spiral Matrix II        
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        
        """
        
        if not n:
            return []
        
        ret=[[0] * n for _ in range(n)]
        
        dr=[0,1,0,-1]
        dc=[1,0,-1,0]
        
        r=0
        c=0
        di=0
        
        for i in range(1,n*n+1):
          ret[r][c]=i
          rc=r+dr[di]
          cc=c+dc[di]
          if 0<=rc<n  and 0<=cc<n and not ret[rc][cc]:
            
            r=rc
            c=cc
            
            print('if',r,c)
            
          else:
            di=(di+1)%4
            r=r+dr[di]
            c=c+dc[di]
            
            print('else',r,c)
          print(ret)
         
        return ret
if __name__ == "__main__":
    print(Solution().generateMatrix(3))          
        
        
        
#60. Permutation Sequence        
class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        if n==0:
            return []
        
        import math
        ret=[]
        nlist=list(range(1,n+1))
        def getdigit(nlist,k,ret,n):
            
            while n>0:
                 temp=int(math.ceil(k*1.0/(math.factorial(n-1)))-1)
                 if temp<0:
                     temp=temp+len(nlist)
                 
                 ret.append(nlist[temp])
                 print(temp,ret,nlist,k)
            
                 k=k%(math.factorial(n-1))
                 n=n-1
                 nlist=nlist[:temp]+nlist[temp+1:]
        
        getdigit(nlist,k,ret,n)
        return ''.join(str(e) for e in ret)
if __name__ == "__main__":
    print(Solution().getPermutation(3,1))         
        
#61. Rotate List        
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
#        Given 1->2->3->4->5->NULL and k = 2,
#
#        return 4->5->1->2->3->NULL.        
        
        if not head:
            return None
        
        p1=p2=head
        cur=head
        n=0
        while cur:
           n+=1
           cur=cur.next
        print(n)    
        k=k%n
        
        if k==0:
            return head
        
        for _ in range(k):
            p2=p2.next
        
        while p2.next:
              p2=p2.next
              p1=p1.next
        newhead=p1.next
        p1.next=None
        p2.next=head
        return newhead
        
        
#62. Unique Paths        
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        if m==0 and n==0:
            return 0
        
        if m==0 or n==0:
            return 1
        
        #making the map
        
        pathmap=[[0]*n for _ in range(m)]
        
        for i in range(n):
            pathmap[0][i]=1
        
        for j in range(m):
            pathmap[j][0]=1
            
        for i in range(1,n):
            for j in range(1,m):
               pathmap[j][i] =pathmap[j-1][i]+pathmap[j][i-1]
            
            
            
        return pathmap[m-1][n-1]
        
        
#63. Unique Paths II       
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        if not obstacleGrid:
            return 0
        
        for i in range(len(obstacleGrid)):
            if obstacleGrid[i][0]==1:
               if i+1<len(obstacleGrid):
                  obstacleGrid[i+1][0]=1
                  
        
               
        for j in range(len(obstacleGrid[0])):
            if obstacleGrid[0][j]==1:
               if j+1<len(obstacleGrid[0]):
                  obstacleGrid[0][j+1]=1
        
        for i in range(len(obstacleGrid)):
            if obstacleGrid[i][0]==1:
               obstacleGrid[i][0]=0
            else:
               obstacleGrid[i][0]=1
        
        
        for j in range(1,len(obstacleGrid[0])):
            if obstacleGrid[0][j]==1:
               obstacleGrid[0][j]=0
            else:
               obstacleGrid[0][j]=1
               
        for i in range(1,len(obstacleGrid)):
            for j in range(1,len(obstacleGrid[0])):
                if obstacleGrid[i][j]==1:
                   obstacleGrid[i][j]=0
                else:                    
                   obstacleGrid[i][j] =obstacleGrid[i][j-1]+obstacleGrid[i-1][j]       
       
               
        return obstacleGrid[len(obstacleGrid)-1][len(obstacleGrid[0])-1]

if __name__ == "__main__":
    print(Solution().uniquePathsWithObstacles([[ 0, 0, 0 ], [ 0, 1, 0 ],[ 0, 0, 0 ]]))        
        
        
        
#64. Minimum Path Sum        
class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m=len(grid)
        n=len(grid[0])
        
        for i in range(1,m):
            grid[i][0]+=grid[i-1][0]

        for j in range(1,n):
            grid[0][j]+=grid[0][j-1]


        for i in range(1,m):
           for j in range(1,n):
               
               grid[i][j]+=min(grid[i-1][j],grid[i][j-1])
           
        return grid[-1][-1]
           
#71. Simplify Path        
class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        places=[p for p in path.split('/') if p!='.' and p!='']   
        
        stack=[]
        for p in places:
            if p=='..':
                if  stack:
                    stack.pop()
            else:
                stack.append(p)
        return '/'+'/'.join(stack)
                    
#73. Set Matrix Zeroes            
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        m=len(matrix)
        if m==0:
             return 
        n=len(matrix[0])
        
        for i in range(m):
            if matrix[i][0]==0:
                FC=0
                break
            else:
                FC=1
                
        for j in range(n):
            if matrix[0][j]==0:
                #print(matrix[0][j])
                FR=0
                break
            else:
                FR=1
            print(FR)      
        for i in range(1,m):
            for j in range(1,n):
                if matrix[i][j]==0:
                    matrix[0][j]=0
                    matrix[i][0]=0  
                   
                    
        for i in range(1,m):
            if matrix[i][0]==0:
                for j in range(n):
                    matrix[i][j]=0
        
        
        for j in range(1,n):
            if matrix[0][j]==0:
                for i in range(m):
                    matrix[i][j]=0
                   
        if not FC:
             
             for i in range(m):
                 
                 matrix[i][0]=0 
        
        if not FR:
             for j in range(n):
                matrix[0][j]=0 
                
#74. Search a 2D Matrix                
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if len(matrix)==0 or len(matrix[0])==0:
            return False
        
        if target<matrix[0][0] or target>matrix[-1][-1]:
            return False
        first=[]
        
        for i in range(len(matrix)):
            first.append(matrix[i][0])
        
        l=0
        r=len(first)-1
        
        while l<=r:
            mid=(l+r)//2
            if first[mid]< target:
                   l=mid+1
            else:
                   r=mid-1
        print(l,r)
        
           
            
        if first[r]==target  or (l<len(first) and first[l]==target):
           return True
        if r<0:
            return False
        
        
        second=matrix[r]
    
        l=0
        r=len(second)-1
        
        while l<=r:
            mid=(l+r)//2
            if second[mid]==target:
                return True
            elif second[mid]< target:
                   l=mid+1
            else:
                   r=mid-1
        
        return False
    
    
        
if __name__ == "__main__":
    print(Solution().searchMatrix([[1]],1))        
                    
            
            
#75. Sort Colors            
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """            
        i=0
        j=0
        for k in range(len(nums)):
            v=nums[k]
            nums[k]=2
            if v<2:
                nums[j]=1
                j+=1
            if v==0:
                nums[i]=0
                i+=1        
            
            
#77. Combinations            
class Solution:
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        listN=list(range(1,n+1))
        result=[]
        results=[]
        def com(listN,m,result,results):
            if m==0:
               results.append(result)               
               return                 
            
            for i in range(len(listN)):
                result+=[listN[i]]
                com(listN[i+1:],m-1,result,results)
        
        com(listN,k,result,results)  
        return results   
if __name__ == "__main__":
    print(Solution().combine(3,1))             
            
        def combine(self, n, k):
            if k==0:
                return [[]]
            return [pre+[i]for i in range(1,n+1) for pre in self.combine(i-1,k-1)]
            
        
#78. Subsets
class Solution:
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def combine(nums,k): 
            if k==0:
               return [[]]
            return [pre+[nums[i]] for i in range(len(nums))  for pre in combine(nums[:i],k-1)]
        
        res=[]
        for i in range(len(nums)+1):
            t=combine(nums,i)
            res+=t
            
        return res
       res=[]
       def dfs(res,index,path,nums):
           res.append(path)
           for i in range(index,len(nums)):
               dfs(res,i+1,path+[nums[i]],nums)
        
       dfs(res,0,[],nums)
 
if __name__ == "__main__":
    print(Solution().subsets([1,2,3]))  

#79. Word Search               
class Solution:
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        
        
        def dfs(board,word,i,j):
            if len(word)==0:
                return True
            if i<0 or i>=len(board) or j<0 or j>=len(board[0]) or board[i][j]!=word[0]:
                return False
            tmp=board[i][j]
            board[i][j]='#'
            res=dfs(board,word[1:],i-1,j) or dfs(board,word[1:],i+1,j) or dfs(board,word[1:],i,j-1) or dfs(board,word[1:],i,j+1)
            board[i][j]=tmp
            return res
         
        if not board:
            return False
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(board,word,i,j):
                    return True
        return False    
             
if __name__ == "__main__":
    print(Solution().exist([['A','B','C','E'],['S','F','C','S'],['A','D','E','E']],'SEE'))                     
        
#80. Remove Duplicates from Sorted Array II                    
class Solution:
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """        
                
        if not nums:
            return 0
        
        i=0
        for n in nums:
            if i<2 or n>nums[i-2]:
                nums[i]=n
                i+=1
        return i
if __name__ == "__main__":
    print(Solution().removeDuplicates([1,1,1,2,2,3]))             


#81. Search in Rotated Sorted Array II               
class Solution:
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        
        
        if not nums:
            return False
        left=0
        right=len(nums)-1
        
        
        
        while left <= right:
         mid=(left+right)  >> 1
         if nums[mid]==target:
            return True
        
         if nums[mid]==nums[left]  and nums[mid]==nums[right]:
            left+=1
            right-=1
        
         elif nums[left] <=nums[mid]:
             if target>= nums[left]  and target <nums[mid]:
                  right=mid-1
             else:
                 left=mid+1
                 
         else:
           
               if target> nums[mid]  and target <= nums[right]:
                   left=mid+1
               else:
                   right=mid-1
        return False
     
            
if __name__ == "__main__":
    print(Solution().search([3,1,1],0))        

    
#82. Remove Duplicates from Sorted List II    
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """    
        dummy=ListNode(0)
        dummy.next=head
        if not head:
            return None
        
        slow=dummy
        fast=head
        
        while fast:
            while fast.next and fast.val==fast.next.val:
                  fast=fast.next
            if slow.next!=fast:
                slow.next=fast.next
                fast=slow.next
                             
            else:
                fast=slow.next
                slow=slow.next
        return dummy.next
                
#86. Partition List                
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        p1=ListNode(0)
        p2=ListNode(0) 
        node1=p1
        node2=p2
        cur=head
        while cur:
            
            if cur.x<x:
               p1.next=cur
               p1=p1.next
            else:
               p2.next=cur
               p2=p2.next
            cur=cur.next
       p1.next=node2.next
       p2.next=None
       return  node1.next       
                
                
#89. Gray Code                
class Solution:
    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        00,01,11,10 -> (000,001,011,010 ) (110,111,101,100).
        """      
        res=[0]  

        for i in range(n):
            res+=[pow(2,i)+x for x in reversed(res)]
        return res
if __name__ == "__main__":
    print(Solution().grayCode(0))                 
                
#90. Subsets II                
class Solution:
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if not nums:
            return [[]]
        
        ret=[[]]   
        for i in range(len(nums)):
            if i==0 or nums[i]!=nums[i-1]:
                l=len(ret)
            for j in range(len(ret)-l,len(ret)):
                ret+=[ret[j]+[nums[i]]]
        return ret
if __name__ == "__main__":
    print(Solution().subsetsWithDup([1,2,2]))
                
#91. Decode Ways             
class Solution:
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        dp=[0]*(len(s)+1)
        dp[0]=1
        for i in range(1,len(s)+1):
            if s[i-1]!='0':
                dp[i]+=dp[i-1]
            if i>1 and s[i-2:i]>'09' and  s[i-2:i]<'27':
               dp[i]+=dp[i-2]
        return dp[len(s)]
 
if __name__ == "__main__":
    print(Solution().numDecodings('0'))

#92. Reverse Linked List II
class Solution:
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        if not head:
            return None
        if m==n:
            return head
        
        dummy=ListNode(0)
        dummy.next=head
        pre=dummy
        
        for i in range(m-1):
            pre=pre.next
            
        NextNode=None
        PreNode=None
        curNode=pre.next
        
        for i in range(n-m+1):
            NextNode=curNode.next
            curNode.next=PreNode
            PreNode=curNode
            curNode=NextNode
        
        pre.next.next=curNode
        pre.next=   PreNode
        return dummy.next

#93. Restore IP Addresses
class Solution:
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        
        """
        def dfs(s,index,path,res):
            if index==4:
                if not s:
                    res.append(path[:-1])
                return 
            for i in range(1,4):
                if i <=len(s):
                    if i==1:
                        dfs(s[i:],index+1,path+s[:i]+'.',res)
                    if i==2 and s[0]!='0':
                        dfs(s[i:],index+1,path+s[:i]+'.',res)
                    if i==3 and s[0]!='0' and int(s[:i])<=255:
                        dfs(s[i:],index+1,path+s[:i]+'.',res)
        res=[]
        dfs(s,0,'',res)
        return res
        
if __name__ == "__main__":
    print(Solution().restoreIpAddresses("25525511135"))

#94. Binary Tree Inorder Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return None
        self.inorderTraversal(root.left)
        print(root.val)
        self.inorderTraversal(root.right)
        
        
        res=[]
        stack=[]
        while True:
            while root:
                stack.append(root)
                root=root.left
                
            if not stack :
                return res
            
            node=stack.pop()
            res.append(node.val)
            root=node.right
            
            
            
            
# Definition for a binary tree node.
class TreeNode:
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None
     def __str__(self):
         return str(self.val)
         
class Solution:
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """     
        if n==0:
            return []
        def genTree(start,end):
            res=[]
            for i in range(start,end+1):
                    for lnode in genTree(start,i-1):
                        for rnode in genTree(i+1,end):
                            root=TreeNode(i)
                            root.left=lnode
                            root.right=rnode
                            res.append(root)
           return res or [None]
        return genTree(1,n)
    

#96. Unique Binary Search Trees                
class Solution:
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        G(n) = G(0) * G(n-1) + G(1) * G(n-2) + … + G(n-1) * G(0) 
        """
       
        if n==1 or n==0:
            return 1
        G=[0]*(n+1)
        G[0]=1
        G[1]=1
        
        for i in range(2,n+1):
            for j in range(i):
                print(i,j,G)
                G[i]+=G[j]*G[i-j-1]
        return G[n] 
if __name__ == "__main__":
    print(Solution().numTrees(2))        
        
#98. Validate Binary Search Tree        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        
        stack=[]
        
        pre=None
        
        while  root or  stack:
            while  root:
                stack.append(root)
                root=root.left
            root=stack.pop()
            if  pre and root.val <=pre.val:
                return False
            pre=root
            root=root.right
        return True
            
#102. Binary Tree Level Order Traversal                
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        
        if not root:
            return []

        from collections import deque
        d=deque() 
        d.append(root)
        res=[]
        while d:
              size=len(d)
              temp=[]
              for i in range(size):
                  
                  node=d.popleft()
                  temp+=[node.val]
                  if node.left:
                     d.append(node.left)
        
                  if node.right:
                     d.append(node.right)
              res.append(temp)
        return res
        
#103. Binary Tree Zigzag Level Order Traversal        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []

        from collections import deque
        d=deque() 
        d.append(root)
        res=[]
        direction=False
        while d:
              size=len(d)
              temp=[]
              for i in range(size):
                  
                  node=d.popleft()
                  temp+=[node.val]
                  if node.left:
                     d.append(node.left)
        
                  if node.right:
                     d.append(node.right)
              if direction :
                  temp[:]=temp[::-1]
              direction= not direction
              res.append(temp)
        return res       
        
#105. Construct Binary Tree from Preorder and Inorder Traversal        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if inorder:
            val=preorder.pop(0)
            index=inorder.index(val)
            root=TreeNode(val)
            root.left=self.buildTree(preorder, inorder[:index])
            root.right=self.buildTree(preorder, inorder[index+1:])
            return root

#106. Construct Binary Tree from Inorder and Postorder Traversal        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if inorder:
           val=postorder.pop()
           #print(inorder,val,postorder)
           index=inorder.index(val)
           root=TreeNode(val)
           root.right=self.buildTree( inorder[index+1:],postorder)
           root.left=self.buildTree( inorder[:index],postorder)
           return root
       
        
        
        
#109. Convert Sorted List to Binary Search Tree    
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if not head:
            return None
        if not head.next:
            return TreeNode(head.val)
        
        fast=head.next.next
        slow=head
        
        while fast and fast.next:
            fast=fast.next.next
            slow=slow.next
        temp=slow.next
        slow.next=None
        root=TreeNode(temp.val)
        root.left=self.sortedListToBST(head)
        root.right=self.sortedListToBST(temp.next)  
        return root
  

           
#113. Path Sum II        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res=[]
        def dfs(node,path,sum,res):
            if not node.left and not node.right and sum==node.val:
                res.append(path+[node.val])
                
            if True:
                
                n=node.val
                if node.left:
                   dfs(node.left,path+[n],sum-n,res)
                if node.right:
                   dfs(node.right,path+[n],sum-n,res)
                
            
        dfs(root,[],sum,res)
        return res

  
       
#114. Flatten Binary Tree to Linked List
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        解答 http://bangbingsyb.blogspot.com/2014/11/leetcode-flatten-binary-tree-to-linked.html
        """
        if not root:
            return 
        left=root.left
        right=root.right
        
        root.left=None
        self.flatten(left)
        self.flatten(right)
        
        root.right=left
        cur=root
        
        while cur.right:
            cur=cur.right
        cur.right=right
        
           
       # iterative
        
       
        cur=root
        
        while cur:
            if cur.left:
                if cur.right:
                    nextN=cur.left
                    while nextN.right:
                          nextN=nextN.right
                    nextN.right=cur.right
                cur.right=cur.left
                cur.left=None
           cur=cur.right

#116. Populating Next Right Pointers in Each Node
# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):


        from collections import deque
        d=deque() 
        d.append(root)
        
        cur=root
        while d:
            size=len(d)
            for i in range(size):
                node=d.popleft()
                if node.left:
                    d.append(node.left)
                if node.right:
                    d.append(node.right)
                    
                if not cur:
                    cur=node
                
                else:
                    cur.next=node
                    cur=cur.next
                    if i==size-1:
                        cur.next=None
                        cur=None

# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
                
        from collections import deque
        d=deque() 
        d.append(root)
        
        cur=root
        while d:
            size=len(d)
            for i in range(size):
                node=d.popleft()
                if node.left:
                    d.append(node.left)
                if node.right:
                    d.append(node.right)
                    
                if not cur:
                    cur=node
                
                else:
                    cur.next=node
                    cur=cur.next
                if i==size-1:
                        cur.next=None
                        cur=None


#120. Triangle
class Solution:
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        
        if not triangle:
            return 0
        
        minlen=triangle[-1]
        for layer in range(len(triangle)-2,-1,-1):
            for i in range(layer+1):
                print(minlen)
                minlen[i]=min( minlen[i], minlen[i+1])+triangle[layer][i]
        return minlen[0]
    
if __name__ == "__main__":
    print(Solution().minimumTotal([[1],[2,3]]))            


#127. Word Ladder
class Solution:
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        if endWord not in wordList:
            return 0
        if endWord == beginWord:
            return 1
        if beginWord in wordList:
           wordList.remove(beginWord)
           
        wordict=set(wordList)
        
        front=set([beginWord])
        back=set([endWord])
        
        length=2
        
        while front:
            
            front=wordict & set([word[:index]+ char+word[index+1:] for word in front
                                  for index in range(len(beginWord))
                                  for char in 'abcdefghijklmnopqrstuvwxyz'])
            if front & back:
                return length
            
            length+=1
            if len( front ) >len( back ):
               
              front,back=back,front
            
            
            wordict-=front
        return 0
    
#129. Sum Root to Leaf Numbers    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """    
        if not root:
            return 0
        
        
        sumN=[]
        def dfs(node,sumN,N):
            if root:
               if not node.left and not node.right:
                  sumN+=[N*10+node.val]
               if node.left:
                  dfs(node.left,sumN,N*10+node.val)
            
               if node.right:
                  dfs(node.right,sumN,N*10+node.val)
            
            
        dfs(root,sumN,0) 
        return sum(sumN)
            
#130. Surrounded Regions            
class Solution:
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        
        if not board or not board[0]:
            return
        m=len(board)
        n=len(board[0])
        save=[(i,j) for i in range(m) for j in range(n) if i in {0,m-1} or j in {0,n-1}]
        while save:
            i,j=save.pop()
            if 0<=i<m and 0<=j<n and board[i][j]=='O':
                board[i][j]='M'
#                print(i,j)
#                print(board)
                save+=(i+1,j),(i-1,j),(i,j+1),(i,j-1)
        #print(board)
        for i in range(m):
            for j in range(n):
                
                if board[i][j]=='M':
                   board[i][j]='O'
                elif board[i][j]=='O':
                     board[i][j]='X'
        return board
board=[["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
if __name__ == "__main__":
    print(Solution().solve(board)) 

#131. Palindrome Partitioning            
class Solution:
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]

        """
        
        return [[s[:i]]+rest for i in range(1,len(s)+1) if s[:i]==s[i-1::-1] for rest in self.partition(s[i:])] or [[]]
            
    
if __name__ == "__main__":
    print(Solution().partition("amanaplanacanalpanama"))             
    t=Solution().partition("amanaplanacanalpanama")  
            
#133. Clone Graph            
# Definition for a undirected graph node
# class UndirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []

class Solution:
    # @param node, a undirected graph node
    # @return a undirected graph node
    def cloneGraph(self, node):  
        
        
        if not node:
            return node
    
        root=UndirectedGraphNode(node.label)
        stack=[node]
        
        visit={}
        visit[node.label]=root
        
        while stack:
            top=stack.pop()
            
            for n in top.neighbors:
                if n.label not in visit:
                   stack.append(n)
                   visit[n.label]=UndirectedGraphNode(n.label)
                
                visit[top.label].neighbors.append(visit[n.label])
        return root
        
        
#134. Gas Station        
class Solution:
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        
        
        
        n=len(gas)
        sum=0
        start=0
        total=0
        
        for i in range(n):
            sum+=gas[i]-cost[i]
            if sum<0:
                total+=sum
                sum=0
                start=i+1
        total+=sum
        if total<0:
            return -1
        else:
            return start
  
if __name__ == "__main__":
    print(Solution().canCompleteCircuit([2,3,1],[3,1,2]))             
       
        
#137. Single Number II        
class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        答案：https://discuss.leetcode.com/topic/17629/the-simplest-solution-ever-with-clear-explanation/18
        """
        b0=0
        b1=0
        for  num in nums:
             b0=(b0^num)&(~b1)
             b1=(b1^num)&(~b0)
        return b0
        
        
#138. Copy List with Random Pointer        
# Definition for singly-linked list with a random pointer.
# class RandomListNode(object):
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None

class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        
       if not head:
           return None
       
       root=RandomListNode(head.label)
       
       cur=head
       copy_cur=root
       
       while cur:
             nextnode=RandomListNode(cur.next.label)
             randomnode=RandomListNode(cur.random.label)
             copy_cur.next= nextnode
             copy_cur.random= randomnode
             
             cur=cur.next
             copy_cur=copy_cur.next
      return root
        
#139. Word Break        
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        d=[False]*len(s)
        
        for i in range(len(s)):
            for w in wordDict:
                if w==s[i-len(w)+1:i+1]  and  (i>=len(w) and d[i-len(w)] or i-len(w)==-1):
                    d[i]=True
        return d[-1]
        
        
    
if __name__ == "__main__":
    print(Solution().wordBreak(s,wordDict))             
        
s="bccdbacdbdacddabbaaaadababadad"
wordDict=["cbc","bcda","adb","ddca","bad","bbb","dad","dac","ba","aa","bd","abab","bb","dbda","cb","caccc","d","dd","aadb","cc","b","bcc","bcd","cd","cbca","bbd","ddd","dabb","ab","acd","a","bbcc","cdcbd","cada","dbca","ac","abacd","cba","cdb","dbac","aada","cdcda","cdc","dbc","dbcb","bdb","ddbdd","cadaa","ddbc","babb"]        


#142. Linked List Cycle II        
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None
        if not head.next:
            return None
            
        
        fast=slow=head
        
        
        while True:
            if not fast.next or not fast.next.next:
               return None
            fast=fast.next.next
            slow=slow.next
            
            if fast==slow:
                break
        
        fast=head
        
        while True:
            if fast==slow:
                break
            fast=fast.next
            slow=slow.next
            
        return slow
        
#143. Reorder List        
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: void Do not return anything, modify head in-place instead.
        """
        import time
        if not head or not head.next or not head.next.next:
            return 
        
        # find the middel point 
        #printNodelist(head)
        fast=slow=head
        while fast and fast.next and fast.next.next:
             fast=fast.next.next
             slow=slow.next
        part2=ListNode(-1)
        part2.next=slow.next
        node=slow.next
        slow.next=None
#        printNodelist(head)
#        printNode(part2)
#        printNode(part2.next)
#        printNode(node)        
#        printNode(slow)
#        printNodelist(node)
         
#        #reverse the second part
        print("-------reverse-------")
        current=node
        prenode=None
        nextnode=None
         
        while current and current.next:
               
               nextnode=current.next
#               if nextnode:
#                  print('nextnode.val',nextnode.val)
#               else:
#                  print('nextnode.val','None')
               
               current.next=nextnode.next
#               if current.next:
#                   print('current.next.val',current.next.val )
#               else:
#                   print('current.next.val',"None" )
#                   
               nextnode.next=part2.next
#               if prenode:
#                  print('prenode.val',prenode.val)
#               else:
#                  print('prenode.val',"None")
#               
               part2.next=nextnode
               
#               if current:
#                  print('current.val',current.val)
#               else:
#                  print('current.val',"None" )
#               time.sleep(5)
               
        #printNodelist(part2.next)
        
#        
#       # merge back
        node1=head
        node2=part2.next
#        
        while node2:
              next1=node1.next
              next2=node2.next
              node1.next=node2
              node2.next=next1
              node1=next1
              node2=next2
        #printNodelist(head)
        return head




#while(node != null && node.next != null){
#			ListNode next = node.next;
#			node.next = next.next;
#			next.next = dummy.next;
#			dummy.next = next;
#
#		}



llist = LinkedList()
# Create a list 10->20->30->40->50->60
for i in range(60, 0, -10):
    llist.push(i)
print("Given linked list")
#llist.printList()

if __name__ == "__main__":
    Solution().reorderList(llist.head)            
        
#144. Binary Tree Preorder Traversal   
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """        
        if not root:
            return []
        stack=[root]
        ret=[]
        while stack:
            node=stack.pop()
            if node:
               ret.append(node.val)
               stack.append(node.right)
               stack.append(node.left)
        return ret
        
        
#147. Insertion Sort List        
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        https://discuss.leetcode.com/topic/8570/an-easy-and-clear-way-to-sort-o-1-space
        """

        if not head:
            return head
        
        dummy=ListNode(0)
        cur=head
        
        pre=dummy

# Tile limit exceed         
#        while cur:
#             next=cur.next
#             
#             
#             while pre.next and pre.next.val<cur.val:
#                   pre=pre.next
#             cur.next=pre.next
#             pre.next=cur
#             pre=dummy
#             cur=next
             
             
             
        while cur:
             next=cur.next
             
             if pre.val>=cur.val:
                pre=dummy
             while pre.next and pre.next.val<cur.val:
                   pre=pre.next
             cur.next=pre.next
             pre.next=cur
             #pre=dummy
             cur=next    
             
             
        return dummy.next
        
  
#148. Sort List       
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return head
        if not head.next:
            return head
        
        #print('----llist.head-----')
        #printNodelist(llist.head)  
        pre=head
        p1=head
        p2=head
        
        while p2 and p2.next:
            pre=p1
            p1=p1.next
            p2=p2.next.next
            
        
        pre.next=None
        
        h1=self.sortList(head)
        h2=self.sortList(p1)
#        print('----hval-----')
#        print(h1.val)
#        print(h2.val)
        return self.merge(h1,h2)

    def merge(self,h1,h2):
        
        l=ListNode(0)
        p=l
#        print('--in merge--')
#        print(h1.val)
#        print(h2.val)
#        print(p.val)
        
        while h1  and h2:
            if h1.val<h2.val:
                p.next=h1
                h1=h1.next
            else:
                p.next=h2
                h2=h2.next
            p=p.next
            #print('p.next',p.next)
            
        if h1:
            p.next=h1
        if h2:
            p.next=h2
        return l.next

llist = LinkedList()
llist.push(1)
llist.push(2)
llist.push(3)
llist.push(4)

              
if __name__ == "__main__":
    printNodelist(Solution().sortList(llist.head))                 
        
#150. Evaluate Reverse Polish Notation       
class Solution:
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        if not tokens:
            return 0
        stack=[]
        
        operators='+-*/'
        
        
        
        for token in tokens:
            if not token in operators:
               stack.append(int(token))
            else:
                
               operand1=stack.pop()
               operand2=stack.pop()
               
               if token=='+':
                   stack.append(operand1+operand2)
               elif token=='-':
                   stack.append(-operand1+operand2)
               elif token=='*':
                   stack.append(operand1*operand2)
               else:
                   if operand1*operand2>0 or operand2%operand1 ==0:
                      stack.append(operand2//operand1)
                   else:
                      stack.append(operand2//operand1+1)
        return stack.pop()
                   
#151. Reverse Words in a String                   
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        if not s:
           return ''
        
        t=s.split()
        return ' '.join(t[::-1])               
if __name__ == "__main__":
    print(Solution().reverseWords("the sky is blue"))                  
               
#152. Maximum Product Subarray               
class Solution:
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
           return 0
        
        imax=nums[0]
        imin=nums[0]
        
        r=imax
        
        for num in nums[1:]:
            if num<0:
                imax,imin=imin,imax
            imin=min(imin*num,num)
            imax=max(imax*num,num)
            
            r=max(r,imax)
        return r
if __name__ == "__main__":
    print(Solution().maxProduct([-2,3,-4]))               
                         
#153. Find Minimum in Rotated Sorted Array                
class Solution:
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        import math
        l=0
        r=len(nums)-1
        
        while l<r:
            mid=(l+r)//2
            
            if nums[l]<nums[r]:
                return nums[l]
            
            if nums[mid]>=nums[l]:
                l=mid+1
            else :
                r=mid
            
                
        return nums[l]
if __name__ == "__main__":
    print(Solution().findMin([1,2]))             
            
#162. Find Peak Element                
class Solution:
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        
        if len(nums)==1:
            return 0
        
        l=0
        r=len(nums)-1
        while l<r:
            mid=(l+r)//2
            if nums[mid]>nums[mid+1]:
                r=mid
            else:
                l=mid+1
        return l

if __name__ == "__main__":
    print(Solution().findPeakElement([1, 2, 3, 1]))                
        
#165. Compare Version Numbers
class Solution:
    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        v1=[int(x) for x in version1.split('.')]
        v2=[int(x) for x in version2.split('.')]
        
        n1=len(v1)
        n2=len(v2)
        n=min(n1,n2)
        for i in range(n):
           if v1[i] >v2[i]:
              return 1
           elif v1[i] < v2[i]:
              return -1
        if n1==n2:
            return 0
        elif n1>n2:
                for i in range(n2,n1):
                    #print(v1[i])
                    if v1[i]!=0:                        
                        return 1
                return 0
        elif n1<n2:
             for i in range(n1,n2):
                    #print(i,v2[i])
                    if v2[i]!=0:
                        return -1
             return  0
           
if __name__ == "__main__":
    print(Solution().compareVersion("1","1.0.1"))              
        
#166. Fraction to Recurring Decimal        
class Solution:
    def fractionToDecimal(self, numerator, denominator):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        """  
        res=''
        if numerator * denominator <0:
            res+='-'
        numerator=abs(numerator)  
        denominator=abs(denominator) 
        if numerator%denominator==0:
            return res+str(numerator//denominator)
        
        table={}
        
        
        res+=str(numerator//denominator)
        res+='.'
        remainder=numerator%denominator
        i=len(res)
        
        while remainder!=0:
              if remainder not in table:
                 table[remainder]=i
              else:
                 i=table[remainder]
                 return res[:i]+'('+res[i:]+')'
              res+=str(remainder*10//denominator)
              remainder=(remainder*10)%denominator
              i+=1
        return res
        
if __name__ == "__main__":
    print(Solution().fractionToDecimal(2,3))           
        
        
#173. Binary Search Tree Iterator       
# Definition for a  binary tree node
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class BSTIterator(object):
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.stack=[]
        x=root
        while x:
            self.stack.append(x)
            x=x.left
        

    def hasNext(self):
        """
        :rtype: bool
        """
        return !len(self.stack)==0
        

    def next(self):
        """
        :rtype: int
        """
        node=self.stack.pop()
        x=node.right
        while x:
            self.stack.append(x)
            x=x.left
        return node.val
        
        

# Your BSTIterator will be called like this:
# i, v = BSTIterator(root), []
# while i.hasNext(): v.append(i.next())        
        
        
#177. Nth Highest Salary
#        CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
#BEGIN
#DECLARE M INT;
#set M=N-1;
#  RETURN (
#      # Write your MySQL query statement below.
#      
#      select distinct e1.Salary from Employee e1 where N-1= (select count(distinct e2.Salary) from Employee e2 where e1.Salary<e2.Salary
#                                                            ) 
#      
#  );
#END
#178. Rank Scores  
# Write your MySQL query statement below
#   select Score,
#   ( select count(*)  from ( select distinct score s from Scores) tmp where s>=Score) rank 
#   
#   from Scores
#   order by Score desc;
#179. Largest Number        
class Solution:
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        if not nums:
           return [] 
        class comparekey(str):
            def __lt__(x,y):
                return x+y>y+x
   
        s=[str(num) for num in nums] 
        s.sort(key=comparekey)   
        

        return ''.join(s) if s[0]  !='0' else '0'      
if __name__ == "__main__":
    print(Solution().largestNumber([3, 30, 34, 5, 9]))        

#180. Consecutive Numbers
    # Write your MySQL query statement below
#select distinct l1.Num as ConsecutiveNums 
#from Logs l1,
#     Logs l2,
#     Logs l3
#     
#     where l1.Id=l2.Id-1 and
#           l2.Id=l3.Id-1 and
#           l1.Num=l2.Num and
#           l2.Num=l3.Num;

#187. Repeated DNA Sequences
class Solution:
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        if not s:
            return []
        
        seen=set()
        output=set()
        
        for i  in range(0,len(s)-9):
            if s[i:i+10] not in seen:
                seen.add(s[i:i+10])
            else:
                output.add(s[i:i+10])
        return list(output)
if __name__ == "__main__":
    print(Solution().findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"))     
        
#192. Word Frequency        
#awk '\\
#{ for (i=1; i<=NF; i++) { ++D[$i]; } }\\
#END { for (i in D) { print i, D[i] } }\\
#' words.txt | sort -nr -k 2  

      
# 194. Transpose File       
# Read from the file file.txt and print its transposed content to stdout.
#awk '
#{
#    for (i = 1; i <= NF; i++) {
#        if(NR == 1) {
#            s[i] = $i;
#        } else {
#            s[i] = s[i] " " $i;
#        }
#    }
#}
#END {
#    for (i = 1; s[i] != ""; i++) {
#        print s[i];
#    }
#}' file.txt        
        
#199. Binary Tree Right Side View        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        
        rightMost={}
        
        stack=[(root,0)]
        maxdepth=-1
        
        while stack:
            node,depth=stack.pop()
            
            if node:
                  maxdepth=max(maxdepth,depth)
                  rightMost.setdefault(depth,node.val)
                  stack.append((node.left,depth+1))
                  stack.append((node.right,depth+1))
        return [rightMost[depth] for depth in range(maxdepth+1)]
    
#200. Number of Islands
class Solution:
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid or not grid[0]:
            return 0
        row=len(grid)
        col=len(grid[0])
        sumN=0
        for i in range(row):
            for j in range(col):
                if grid[i][j]=='1':
                    
                   sumN+=self.isConnect(i,j,grid)
                   
        return sumN
        
    def isConnect(self,r,c,grid):
            if 0<=r<len(grid) and 0<=c<len(grid[0]) and grid[r][c]=='1':
               grid[r][c]='#'
               #print(grid)
               self.isConnect(r+1,c,grid)
               self.isConnect(r-1,c,grid)
               self.isConnect(r,c+1,grid)
               self.isConnect(r,c-1,grid)
               return 1
            return 0
            
        
        
        #return sum(sink(i, j) for i in range(len(grid)) for j in range(len(grid[i])))
     
grid=[["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]
if __name__ == "__main__":
    print(Solution().numIslands(grid))        
#201. Bitwise AND of Numbers Range 
class Solution:
    def rangeBitwiseAnd(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        i=0
        while m!=n:
            i+=1
            m>>=1
            n>>=1
            
        return n<<i
        
#207. Course Schedule        
class Solution:
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        def dfs(graph,visited,i):
            if visited[i]==-1:
                return False
            if visited[i]==1:
                return True
            visited[i]=-1
            
            for j in graph[i]:
                if not dfs(graph,visited,j):
                    return False
            visited[i]=1
            return True
            
            
            
        graph=[[]  for _ in range(numCourses)]
        visited=[0]*numCourses
        for pair in prerequisites:
            x,y=pair
            graph[x].append(y) 
        for i in range(numCourses):
            if not dfs(graph,visited,i):
                return False
        return True
            
#208. Implement Trie (Prefix Tree)   
class TrieNode:
    def __init__(self):
        from collections import defaultdict
        self.children=defaultdict(TrieNode)
        self.is_word=False
        
       
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root=TrieNode()
        

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        current=self.root
        for letter in word:
            current=current.children[letter]
        current.is_word=True
        

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        current=self.root
        for letter in word:
            current=current.children.get(letter)
            if not current:
                return False
        return current.is_word
        

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        current=self.root
        for letter in prefix:
            current=current.children.get(letter)
            if not current:
                return False
        return True
        


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)            
        
#209. Minimum Size Subarray Sum        
class Solution:
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
#        #O(N)
#        if not nums:
#            return 0
#        left=total=0
#        result=len(nums)+1  
#
#        for right,n in enumerate(nums):
#            total+=n
#            while total>=s:
#                  result=min(result,right-left+1)
#                  total-=nums[left]
#                  left+=1
#                  print(total,left)
#        return result if result<=len(nums) else 0
    
        #O(Nlog(N))
        def search_left(right,left,nums,target,n):
            
            while left<right:
                mid=(right+left)//2
                if n-nums[mid]>=target:
                    left=mid+1
                else:
                    right=mid
            return left
        
        if not nums:
            return 0
        
        result=len(nums)+1
        
        for idx, n in enumerate(nums[1:],1):
            nums[idx]=nums[idx-1]+n
        
        left=0
        for right,n in enumerate( nums):
            if n>=s:
               left=search_left(right,left,nums,s,n)
               result=min(result,right-left+1)
        return result if result<=len(nums) else 0
                
   
if __name__ == "__main__":
    print(Solution().minSubArrayLen(7,[2,3,1,2,4,3]))               
#210. Course Schedule II 
class Solution:
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        import queue  
        from collections import defaultdict 
        
        g={i:set() for  i in range(numCourses) }
        g_inv=defaultdict(set)
        
        for i , j in prerequisites:
            g[i].add(j)
            g_inv[j].add(i)
        
        q=queue.Queue()
        
        for course in g:
            if not g[course]:
                q.put(course)
        
        topo_sort=[]
        while not q.empty():
            node=q.get()
            topo_sort.append(node)
            for i in g_inv[node]:
                g[i].remove(node)
                if not g[i]:
                    q.put(i)
        return   topo_sort if len(topo_sort)==numCourses else []
        
#211. Add and Search Word - Data structure design    
        
class TrieNode(object):  
      def __init__(self):
        self.word=False
        self.children={}
    
class WordDictionary:
    def __init__(self):
        """
        Initialize your data structure here.
        
        """
        self.root=TrieNode()
        

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: void
        """
        node=self.root
        for c in word:
            if c not in node.children:
                node.children[c]=TrieNode()
            node=node.children[c]
        node.word=True

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        def searchFrom(node,word):
            for i in range(len(word)):
                c=word[i]
                if c=='.':
                    for k in node.children:
                        if searchFrom(node.children[k],word[i+1:]):
                           return True
                    return False
                elif c not in node.children:
                    return False
                
                node=node.children[c]
            return node.word
            
        return searchFrom(self.root,word)

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)        
        
#213. House Robber II        
class Solution:
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)==0:
            return 0
        if len(nums)==1:
            return nums[0]
        if len(nums)==2:
            return max(nums)
        
        def rob0(nums):
            curmax=0
            premax=0
            for i in range(len(nums)-1):
                t=curmax
                curmax=max(premax+nums[i],curmax)
                premax=t
            return curmax
        
        def rob1(nums):
            curmax=0
            premax=0
            for i in range(1,len(nums)):
                t=curmax
                curmax=max(premax+nums[i],curmax)
                premax=t
            return curmax       
        return max(rob0(nums),rob1(nums))
    
#215. Kth Largest Element in an Array        
class Solution:
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        import time 
        
        def partition(nums,lo,hi):
            
            i=lo
            j=hi
            
            while True:
#                print(lo,hi,i,j)
#                time.sleep(5)
                while i <hi and nums[i]<=nums[lo]:
                      i+=1
                while j >lo and nums[j]>nums[lo]:
                      j-=1      
                if i>=j:
                    break
                nums[i] , nums[j] = nums[j] , nums[i]
           
            nums[lo] , nums[j] = nums[j] , nums[lo]
            return j
        
        
        k=len(nums)-k    
        lo=0
        hi=len(nums)-1
        #choose nums[lo] as pivot
        
        while lo < hi:
              j=partition(nums,lo,hi)
              #print(j)
              if j<k:
                 lo=j+1 
              elif j>k:
                 hi=j-1
              else:
                 break
        return nums[k]            
if __name__ == "__main__":
    print(Solution().findKthLargest([3,3,3,3,3,3,3,3,3],1))           
        
        
#216. Combination Sum III        
class Solution:
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        if n>=k*9:
            return [[]]
        
        
        def dfs(k,arr,cur,n,res):
            
            if len(arr)==k:
                if sum(arr)==n:
                    #print(arr)
                    res.append(list(arr))
                    
                return 
                    
            if len(arr)>k or cur>9:
                return 
            for i in range(cur,10):
                arr.append(i)
                dfs(k,arr,i+1,n,res)
                arr.pop()
        res=[]
        dfs(k,[],1,n,res)  
        return res
if __name__ == "__main__":
    print(Solution().combinationSum3(3,7))           
                
        
        
        
class Solution:
    def combinationSum3(self, k, n):
        if n>=k*9:
            return [] 
        def dfs(k,n,cap):
            if k==0 and n==0:
               return [[] ]
            
            return [comb+[last]
                for last in range(1,cap)
                for comb in dfs(k-1,n-last,last)               
                
                ]
            
        return dfs(k,n,10)
if __name__ == "__main__":
    print(Solution().combinationSum3(3,3))         
        
#220. Contains Duplicate III       
class Solution:
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """        
        if  t<0:
            return False
        
        d={}
        
        m=t+1
        for i in range(len(nums)):
            if nums[i]//m in d:
                return True
            if nums[i]//m-1 in d and abs(d[nums[i]//m-1]-nums[i])<=t:
                return True
            if nums[i]//m+1 in d and abs(d[nums[i]//m+1]-nums[i])<=t:
                return True
            
            
            d[nums[i]//m]=nums[i]
            if i-k>=0:
                del  d[nums[i-k]//m]
        return False
if __name__ == "__main__":
    print(Solution().containsNearbyAlmostDuplicate(3,3))        
        
#221. Maximal Square        
class Solution:
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if not len(matrix):
            return 0
        
        m=len(matrix)
        n=len(matrix[0])
        dp=[  [1 if matrix[i][j]=='1' else 0 for j in range(n) ] for i in range(m)]
        
        
        for i in range(1,m):
            for j in range(1,n):
                if matrix[i][j]=='1':
#                    print(i,j)
#                    print(dp[i-1][j])
#                    print(dp[i][j-1])
#                    print(dp[i-1][j-1])
                    dp[i][j]=min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1
                    
                else:
                    dp[i][j]=0
        maxL=max([max(i) for i in dp])
        return maxL*maxL
    
    
matrix=[["0","0","0","1"],["1","1","0","1"],["1","1","1","1"],["0","1","1","1"],["0","1","1","1"]]
if __name__ == "__main__":
    print(Solution().maximalSquare(matrix))         
       
#222. Count Complete Tree Nodes        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """        
#Nice solution, add a word, comparing the depth between left sub tree and right sub tree, 
#If it is equal, it means the left sub tree is a perfect binary tree, not only a full binary tree.
# If it is not , it means the right sub tree is a perfect binary tree.        
 
       if not root:
           return 0
       def depth(root):
           if not root:
               return 0
           return depth(root.left)+1
       
       leftDepth=depth(root.left)
       rightDepth=depth(root.right)
       
       if leftDepth==rightDepth:
          return pow(2,leftDepth)+self.countNodes(root.right)
       else:
          return pow(2,rightDepth)+self.countNodes(root.left)
       
#223. Rectangle Area       
class Solution:
    def computeArea(self, A, B, C, D, E, F, G, H):
        """
        :type A: int
        :type B: int
        :type C: int
        :type D: int
        :type E: int
        :type F: int
        :type G: int
        :type H: int
        :rtype: int
        """
        R1_X=set(range(A,C+1))
        R1_Y=set(range(B,D+1))
        R2_X=set(range(E,G+1))
        R2_Y=set(range(F,H+1))
        
         
        R1_R2_X=R1_X & R2_X
        R1_R2_Y=R1_Y & R2_Y
         
        if len( R1_R2_X ) <2 or len(R1_R2_Y) < 2:
             return (len(R1_X)-1)*(len(R1_Y)-1)+(len(R2_X)-1)*(len(R2_Y)-1)
        else:
             return (len(R1_X)-1)*(len(R1_Y)-1)+(len(R2_X)-1)*(len(R2_Y)-1)-(len(R1_R2_X)-1)*(len(R1_R2_Y)-1)
if __name__ == "__main__":
    print(Solution().computeArea(-2,-2,2,2,-2,-2,2,2))  
if __name__ == "__main__":
    print(Solution().computeArea(0,0,0,0,-1,-1,1,1))          
if __name__ == "__main__":
    print(Solution().computeArea(-2,-2,2,2,3,3,4,4))                      
if __name__ == "__main__":
    print(Solution().computeArea(0,0,50000,40000,0,0,50000,40000) )                     
                        
#227. Basic Calculator II       
class Solution:
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        sign='+'
        stack=[]
        num=0
        
        for i in range(len(s)):
            if s[i].isdigit():
                num=num*10+ord(s[i])-ord('0')
                
            if (not s[i].isdigit() and not s[i].isspace()) or i==len(s)-1:
                
                if sign=='+':
                    stack.append(num)
                elif sign=='-':
                    stack.append(-num)
                elif sign=='*':
                    stack.append(stack.pop()*num)
       
                else:
                    temp=stack.pop()
                    if temp//num<0 and temp%num!=0:
                        stack.append(temp//num+1)
                    else:
                        stack.append(temp//num)
                sign=s[i]
                num=0
                
        return sum(stack)
    
if __name__ == "__main__":
    print(Solution().calculate("3+2*2") )        
        
#228. Summary Ranges        
class Solution:
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        if not nums:
            return []
        if len(nums)==1:
            return [str(nums[0])]
        
        
        res=[]
        
        start=nums[0]
        for i in range(len(nums)-1):
            
            if  nums[i]!=nums[i+1]-1:
                end=nums[i]
                if start==end:
                   res.append(str(start))
                   start=nums[i+1]
                else:
                   res.append(str(start)+'->'+str(end))
                   start=nums[i+1]
            
            if i==len(nums) -2 and nums[i]!=nums[i+1]-1:
               res.append(str(start))
            if i==len(nums) -2 and nums[i]==nums[i+1]-1:
               res.append(str(start)+'->'+str(nums[i+1]))
        return res
if __name__ == "__main__":
    print(Solution().summaryRanges( [0,2,3,4,6,8,9]) )          
        
        
#229. Majority Element II        
class Solution:
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return []
        
        count1=0
        candidate1=-1
        count2=0
        candidate2=0
        
        for u in nums:
            if u==candidate1:
                count1+=1
            elif u==candidate2:
                count2+=1
            elif count1==0:
                candidate1,count1=u,1
            elif count2==0:
                candidate2,count2=u,1
            else:
                count2-=1
                count1-=1
        return [n for n in (candidate1,candidate2) if nums.count(n) > len(nums)//3]
if __name__ == "__main__":
    print(Solution().majorityElement( [0,0,0]) )                  
        
        
#230. Kth Smallest Element in a BST        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        count=[]
        def inOrder(node,count):
            if not node:
                return 
            if node.left:
                inOrder(node.left,count)
            count.append(node.val)
            if node.right:
                inOrder(node.right,count)
        inOrder(root,count)
        return count[k-1]
        
        
#236. Lowest Common Ancestor of a Binary Tree        
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return root
        
        if p==root:
            return p
        
        if q==root:
            return q
        
        
        left=self.lowestCommonAncestor(root.left,p,q)
        right=self.lowestCommonAncestor(root.right,p,q)
        
        if not left:
            return right
        if not right:
            return left
        
        if left!=right:
            return root
            
#238. Product of Array Except Self        
class Solution:
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res=[1]*len(nums)
        
        for i in range(1,len(nums)):
            res[i]=nums[i-1]*res[i-1]
        print(res)
        
        right=1
        for j in range(len(nums)-1,-1,-1):
            res[j]=res[j]*right
            right=right*nums[j]
            print(right)
        return res
if __name__ == "__main__":
    print(Solution().productExceptSelf( [1,2,3,4]) )              
        
#240. Search a 2D Matrix II        
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if len(matrix)==0  or len(matrix[0]==0:
            return False
        
        height=len(matrix)
        width=len(matrix[0])
        
        col=0
        row=height-1
        
        while col<width and row>=0:
            if matrix[row][col] > target:
                row-=1
            elif matrix[row][col] < target:
                 col+=1
            else:
                return True
        return False
#241. Different Ways to Add Parentheses                
class Solution:
    def diffWaysToCompute(self, input):
        """
        :type input: str
        :rtype: List[int]
        """

        
        def compute(string):
            
            if not string:
               return []
            
            if string.isdigit():
               return [int(string)]
            res=[]
            for i, c in enumerate(string):
                if c in '+-*':
                    left=compute(string[:i])
                    right=compute(string[i+1:])
                    print(left,right,res)
                    
                    for m in left:
                        for n in right:
                            print(res)
                            if c=='+':                                
                               res.append(m+n)
                            elif c=='-':                                
                               res.append(m-n)
                            elif c=='*':                                
                               res.append(m*n)
            return res
        
        
        return compute(input)
if __name__ == "__main__":
    print(Solution().diffWaysToCompute( "2-1-3") )                   
                    
#260. Single Number III            
class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        diff=0
        for num in nums:
            diff^=num
        
        
        #get the last set of 1
        diff&=~(diff-1)
        c1=0
        c2=0
        for num in nums:
            if diff&num==0:
                c1^=num
            else:
                c2^=num
        return [c1,c2]
if __name__ == "__main__":
    print(Solution().singleNumber( [1, 2, 1, 3, 2, 5] )  )           

#264. Ugly Number II                
class Solution:
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """                
                
        
        if n <= 0:
            return 0
        
        if n==1:
            return 1
        
        p2=0
        p3=0
        p5=0
        
        res=[0]*n
        res[0]=1
        for i in range(1,n):
            print(res[p2],res[p3],res[p5])
            res[i]=min(res[p2]*2,res[p3]*3,res[p5]*5)
            
            if res[i] == res[p2]*2:
                p2+=1
            if res[i] == res[p3]*3:
                p3+=1
            if res[i] == res[p5]*5:
                p5+=1
            
            
        return res[-1]
        
if __name__ == "__main__":
    print(Solution().nthUglyNumber( 8)  )         
        
#274. H-Index        
class Solution:
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        if not    citations:
           return 0
        
        L=len(citations)
        res=[0]*(L+1)
        for citation in citations:
             if citation >L:
                 res[L]+=1
             else:
                 res[citation]+=1
        
        ci=0
        for h in range(L,-1,-1):
            ci+=res[h]
            if ci>=h:
                return h
if __name__ == "__main__":
    print(Solution().hIndex( [3, 0, 6, 1, 5]  ))            
        
#275. H-Index II        
class Solution:
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        if not    citations:
           return 0

        L=len(citations)
        res=0
        for i in range(L-1,-1,-1):
            if L-i<=citations[i]:
                res=max( L-i,res)
        return res
            
if __name__ == "__main__":
    print(Solution().hIndex( sorted([3, 0, 6, 1, 5] ) ))            
                    
            
#279. Perfect Squares        
class Solution:
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n<=1:
            return 1
        lst=[]
        i=1
        while i*i<=n:
            lst.append(i*i)
            i+=1
            
            
        toCheck={n}
        
        cnt=0
        while toCheck:
              cnt+=1
              temp=set()
              for x in toCheck:
                  for y in lst:
                      if x==y:
                          return cnt
                      if x<y:
                          break
                      temp.add(x-y)
              
              toCheck=temp
        return cnt
if __name__ == "__main__":
    print(Solution().numSquares( 13 ))               
#284. Peeking Iterator        
# Below is the interface for Iterator, which is already defined for you.
#
# class Iterator(object):
#     def __init__(self, nums):
#         """
#         Initializes an iterator object to the beginning of a list.
#         :type nums: List[int]
#         """
#
#     def hasNext(self):
#         """
#         Returns true if the iteration has more elements.
#         :rtype: bool
#         """
#
#     def next(self):
#         """
#         Returns the next element in the iteration.
#         :rtype: int
#         """

class PeekingIterator(object):
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iter=iterator
        self.temp=self.iter.next() if self.iter.hasNext() else None

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.temp

    def next(self):
        """
        :rtype: int
        """
        ret=self.temp
        self.temp=self.iter.next() if self.iter.hasNext() else None
        return ret

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.temp is not None
        

# Your PeekingIterator object will be instantiated and called as such:
# iter = PeekingIterator(Iterator(nums))
# while iter.hasNext():
#     val = iter.peek()   # Get the next element but not advance the iterator.
#     iter.next()         # Should return the same value as [val].        
        
#287. Find the Duplicate Number       
class Solution:
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
                
        if not nums:
            return 0
        slow=nums[0]
        fast=nums[nums[0]]
        
        while slow!=fast:
              slow=nums[slow]
              fast=nums[nums[fast]]
        
        
        fast=0
        
        while slow!=fast:
              slow=nums[slow]
              fast=nums[fast]
        return slow
        
#289. Game of Life
class Solution:
    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        #0 dead,1 live  2 dead -> live ; 3 live -> dead

        if len(board)==0  or len(board[0])==0:
            return 
        
        M= len(board)
        N=len(board[0])
        for iM in range(M):
            for iN in range(N):
                neighbor=sum(board[i][j]%2 for i in range(iM-1,iM+2) for j in range(iN-1,iN+2) if 
                             0<=i<M and 0<=j<N) -board[iM][iN]
                if board[iM][iN]==0 and neighbor==3:
                   board[iM][iN]=2
                elif board[iM][iN]==1 and (neighbor<2 or neighbor>3):
                   board[iM][iN]=3
                
        for iM in range(M):
            for iN in range(N):
                if board[iM][iN]==2:
                   board[iM][iN]=1
                if board[iM][iN]==3:
                   board[iM][iN]=0
                
#299. Bulls and Cows                
class Solution:
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
        if not secret :
           return ''

        res=''
        
        L=len(secret)
        sh={}
        gh={}
        bull=0
        for i in range(L):
            if secret[i]==guess[i]:
               bull+=1
            else:
                sh[secret[i]]=sh.get(secret[i],0)+1
                gh[guess[i]]=gh.get(guess[i],0)+1
        cow=0
        for k in sh:
            if k in gh:
                cow+=min(sh[k],gh[k])
        print(sh)
        print(gh)
        return '{0}A{1}B'.format(bull,cow)

if __name__ == "__main__":
    print(Solution().getHint( "1807","7810" ))                
                
if __name__ == "__main__":
    print(Solution().getHint( "1123","0111" ))                
                
                
#300. Longest Increasing Subsequence                
class Solution:
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """                
        if not nums:
            return 0
        
        dp=[0]*len(nums)
        ans=1
        dp[0]=1
        print(nums)
        for i in range(1,len(nums)):
            temp=1
            for j in range(i):
                
                if nums[i]>nums[j]:
                    #print(temp,dp[j])
                    temp=max(temp,dp[j]+1)
                #print(i,j,dp,nums[i],nums[j],temp)
            dp[i]=temp
            ans=max(ans,dp[i])
        return ans
    
        if not nums:
            return 0
        
        size=0
        tail=[0]*len(nums)
        
        
        
        for x in nums:
            i=0
            j=size
            
            while i!=j:
                m=(i+j)//2
                if tail[m]<x:
                    i=m+1
                else:
                    j=m
            tail[i]=x
            size=max(i+1,size)
        return size
 
    
if __name__ == "__main__":
    print(Solution().lengthOfLIS( [10, 9, 2, 5, 3, 7, 101, 18] ))  
#304. Range Sum Query 2D - Immutable 
class NumMatrix:
    # Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)  

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        m=len(matrix)
        n=len(matrix[0])
        
        if m==0 or n==0:
            return 0
        self.dp=[[0  for _ in range(n+1)]  for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                self.dp[i][j+1]=self.dp[i][j]+matrix[i][j]
 

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        summ=0
        
        for r in range(row1,row2+1):
            
                summ+=self.dp[r][col2+1]-self.dp[r][col1]
                
        return summ
    
matrix=[[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]    
obj = NumMatrix([[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]])
param_1 = obj.sumRegion(*[2,1,4,3])

if __name__ == "__main__":
    print(Solution().lengthOfLIS( [10, 9, 2, 5, 3, 7, 101, 18] ))              
        
["NumMatrix","sumRegion","sumRegion","sumRegion"]
[[[[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]],[2,1,4,3],[1,1,2,2],[1,2,2,4]]

#306. Additive Number        
class Solution:
    def isAdditiveNumber(self, num):
        """
        :type num: str
        :rtype: bool
        """  
        def isAdditive(remaining,num1,num2):
            if not remaining:
                return True
            c=str(int(num1)+int(num2))
            print(num1,num2,c,remaining)
            if not remaining.startswith( c ):
                return False
            num1,num2=num2,c
            remaining=remaining[len(c):]
            return isAdditive(remaining,num1,num2)
        L=len(num)
        
        if L==0:
            return False
        
        for i in range(1,(L+1)//2):
            if num[0]=='0' and i>=2:
                break
            
            for j in range(i+1,L):
                if L-j<i or L-j<j-i:
                    break
                if num[i]=='0' and j-i>=2:
                    break
                num1=num[0:i]
                num2=num[i:j]
                remaining=num[j:]
                #print(num1,num2,remaining)
                #print(isAdditive(remaining,num1,num2))
                if isAdditive(remaining,num1,num2):
                    return True
        return False
    
if __name__ == "__main__":
    print(Solution().isAdditiveNumber( "123" ))          
        
        
        
def isAdditive(remaining,num1,num2):
            if not remaining:
                return True
            c=str(int(num1)+int(num2))
            print(num1,num2,c,remaining)
            if not remaining.startswith( c ):
                return False
            num1,num2=num2,c
            remaining=remaining[len(c):]
            return isAdditive(remaining,num1,num2)

               
isAdditive("2358",'1','1')              
                
#307. Range Sum Query - Mutable  
class Node(object):
    
    def __init__(self, start,end):
        self.start=start
        self.end=end
        self.total=0
        self.left=None
        self.right=None
              
class NumArray:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        def buildtree(nums,l,r):
            if l>r:
                return None
            if l==r:
                n=Node(l,r)
                n.total=nums[l]
                return n
            
            mid=(l+r)//2
            
            root=Node(l,r)
            
            root.left= buildtree(nums,l,mid)
            root.right= buildtree(nums,mid+1,r)
            
            root.total=root.left.total+root.right.total
            
            return root
                
        
        
        self.root=buildtree(nums,0,len(nums)-1)

    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: void
        """
        
        def updateVal(root,i,val):
            if root.start==root.end:
                root.total=val
                return val
                
            mid=(root.start+root.end)//2
            
            if i<=mid:
                updateVal(root.left,i,val)
            else:
                updateVal(root.right,i,val)
                
            root.total=root.left.total+root.right.total
            
            return root.total
        return updateVal(self.root,i,val)
            
           
                

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        def Rangesum(root,i,j):
            if root.start==i and root.end==j:
                return root.total
            
            mid=(root.start+root.end)//2
            
            if j<=mid:
                return Rangesum(root.left,i,j)
            elif i>mid:
                return Rangesum(root.right,i,j)
            else:
                return Rangesum(root.left,i,mid)+Rangesum(root.right,mid+1,j)
        return Rangesum(self.root, i, j)


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(i,val)
# param_2 = obj.sumRange(i,j)                
                
#309. Best Time to Buy and Sell Stock with Cooldown                
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices)<2:
           return 0
       
        buy=-prices[0]
        sell=0
        prev_buy=0
        prev_sell=0
        
        for price in prices:
            prev_buy=buy
            buy=max(prev_buy,prev_sell-price)
            prev_sell=sell
            sell=max(prev_sell,prev_buy+price)
            print(price,buy,sell)
        return sell
if __name__ == "__main__":
    print(Solution().maxProfit( [1, 2, 3, 0, 2] ))                    
                
                
def maxProfit(prices):
    if len(prices) < 2:
        return 0
    sell, buy, prev_sell, prev_buy = 0, -prices[0], 0, 0
    for price in prices:
        prev_buy = buy
        buy = max(prev_sell - price, prev_buy)
        prev_sell = sell
        sell = max(prev_buy + price, prev_sell)
        print(price,buy,sell)
    return sell                
maxProfit( [1, 2, 3, 0, 2]  )              
                
#310. Minimum Height Trees                
class Solution:
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        if n==0:
            return []
        if n==1:
            return 0
        
        adj=[set() for _ in range(n)]
        
        
        
        for i,j in edges:
            adj[i].add(j)
            adj[j].add(i)
        leaves=[i for i  in range(n) if len(adj[i]==1)]
        
        
        while n>2:
            n-=len(leaves)
            newleaves=[]
            
            for i in leaves:
                j=adj[i].pop()
                adj[j].remove(i)
                if len(adj[j])==1:
                    newleaves.append(j)
            
            leaves=newleaves
        return leaves
            
#313. Super Ugly Number            
class Solution:
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        [2, 7, 13, 19]
        """            
        if n <= 0:
            return 0
        
        if n==1:
            return 1
        res=[0  for _ in range(n)]
        res[0]=1
        idx=[0 for _ in range(len(primes))]
        for i in range(1,n):
            minN=2**32
            minIndex=0
            
            for j in range(len(primes)):
                  if primes[j]*res[idx[j]] <minN:
                      minN=primes[j]*res[idx[j]]
                      minIndex=j
                  elif primes[j]*res[idx[j]]==minN:
                       idx[j]+=1
            res[i]=minN
            idx[minIndex]+=1
            
        return res[-1]
if __name__ == "__main__":
    print(Solution().nthSuperUglyNumber(100000, [7,19,29,37,41,47,53,59,61,79,83,89,101,103,109,127,131,137,139,157,167,179,181,199,211,229,233,239,241,251] ))              

#318. Maximum Product of Word Lengths            
class Solution:
    def maxProduct(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        if len(words)  ==0:
            return 0
        
        value=[0 for _ in range(len(words))]
        
        
        for i in range(len(words)):
            for j in range(len(words[i])):
                value[i] |=1<<(ord(words[i][j])-ord('a'))
        
        
        maxlength=0
        for i in range(len(words)-1):
            for j in range(i+1,len(words)):
                if value[i] & value[j] ==0 and len(words[i]) * len(words[j]) > maxlength: 
                   maxlength=len(words[i]) * len(words[j]) 
        return maxlength
if __name__ == "__main__":
    print(Solution().maxProduct(["abcw", "baz", "foo", "bar", "xtfn", "abcdef"] ))              
        
#319. Bulb Switcher        
class Solution:
    def bulbSwitch(self, n):
        """
        :type n: int
        :rtype: int
        """
#        if n==0 or n==1 :
#            return n
#        bulb=[ True for _ in range(n+1)]
#        for i in range(2,n+1):
#            for j in range(i,n+1):
#                if j%i==0:
#                   bulb[j]= not  bulb[j]
        return int(n**0.5)
if __name__ == "__main__":
    print(Solution().bulbSwitch(9999 ))     


         
#322. Coin Change            
class Solution:
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        if amount==0:
            return 0
        
        if not coins:
            return 0
        
        visited=[ False for _ in range(amount+1)]
        visited[0]=True
        
        value1=[0]
        value2=[]
        count=0
        while value1:
            count+=1
            for value in value1:
                for coin in coins:
                    newvalue=coin+value
                    if newvalue==amount:
                        return count
                    elif newvalue>amount:
                        continue
                    elif not visited[newvalue]:
                        visited[newvalue]=True
                        value2.append(newvalue)
            value1,value2=value2,[] 
        return -1
if __name__ == "__main__":
    print(Solution().coinChange([1, 2, 5], 11))                             
        
        
#324. Wiggle Sort II        
class Solution:
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        def findKthsmallest(nums, k):
            def partition(nums,lo,hi):
                i=lo
                j=hi
                while True:
                  while i <hi and nums[i]<=nums[lo]:
                      i+=1
                  while j >lo and nums[j]>nums[lo]:
                      j-=1      
                  if i>=j:
                    break
                  nums[i] , nums[j] = nums[j] , nums[i]
                nums[lo] , nums[j] = nums[j] , nums[lo]
                return j
            k=len(nums)-k   
            lo=0
            hi=len(nums)-1
            while lo < hi:
              j=partition(nums,lo,hi)
              #print(j)
              if j<k:
                 lo=j+1 
              elif j>k:
                 hi=j-1
              else:
                 break
            return nums[k]    
                
        
        n=len(nums)
        m=n//2+1
        median=findKthsmallest(nums, m)
        #nums2=nums[:]
        i=0
        j=0
        k=n-1
        while j<=k:
             if nums[j]<median:
               nums[i],nums[j]=nums[j],nums[i]
               i+=1
               j+=1
               
             elif nums[j]>median:
                  nums[k],nums[j]=nums[j],nums[k]
                  k-=1
        
             else:
                 j+=1
                 
        half = len(nums[::2])
        nums[::2], nums[1::2] = nums[:half][::-1], nums[half:][::-1]
        #print(nums,nums2)
#        a=0
#        b=1
        
#        m=(n+1)//2
#        for i in range(m-1,-1,-1):
#            nums[a]=nums2[i]
#            a+=2
#        
#        for i in range(n-1,m-1,-1):
#            #print(b,i)
#            nums[b]=nums2[i]
#            b+=2
        return nums
if __name__ == "__main__":
   t=Solution().wiggleSort([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768,769,770,771,772,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,799,800,801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856,857,858,859,860,861,862,863,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,976,977,978,979,980,981,982,983,984,985,986,987,988,989,990,991,992,993,994,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1044,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069,1070,1071,1072,1073,1074,1075,1076,1077,1078,1079,1080,1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200,1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1328,1329,1330,1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1356,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1391,1392,1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408,1409,1410,1411,1412,1413,1414,1415,1416,1417,1418,1419,1420,1421,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440,1441,1442,1443,1444,1445,1446,1447,1448,1449,1450,1451,1452,1453,1454,1455,1456,1457,1458,1459,1460,1461,1462,1463,1464,1465,1466,1467,1468,1469,1470,1471,1472,1473,1474,1475,1476,1477,1478,1479,1480,1481,1482,1483,1484,1485,1486,1487,1488,1489,1490,1491,1492,1493,1494,1495,1496,1497,1498,1499,1500,1501,1502,1503,1504,1505,1506,1507,1508,1509,1510,1511,1512,1513,1514,1515,1516,1517,1518,1519,1520,1521,1522,1523,1524,1525,1526,1527,1528,1529,1530,1531,1532,1533,1534,1535,1536,1537,1538,1539,1540,1541,1542,1543,1544,1545,1546,1547,1548,1549,1550,1551,1552,1553,1554,1555,1556,1557,1558,1559,1560,1561,1562,1563,1564,1565,1566,1567,1568,1569,1570,1571,1572,1573,1574,1575,1576,1577,1578,1579,1580,1581,1582,1583,1584,1585,1586,1587,1588,1589,1590,1591,1592,1593,1594,1595,1596,1597,1598,1599,1600,1601,1602,1603,1604,1605,1606,1607,1608,1609,1610,1611,1612,1613,1614,1615,1616,1617,1618,1619,1620,1621,1622,1623,1624,1625,1626,1627,1628,1629,1630,1631,1632,1633,1634,1635,1636,1637,1638,1639,1640,1641,1642,1643,1644,1645,1646,1647,1648,1649,1650,1651,1652,1653,1654,1655,1656,1657,1658,1659,1660,1661,1662,1663,1664,1665,1666,1667,1668,1669,1670,1671,1672,1673,1674,1675,1676,1677,1678,1679,1680,1681,1682,1683,1684,1685,1686,1687,1688,1689,1690,1691,1692,1693,1694,1695,1696,1697,1698,1699,1700,1701,1702,1703,1704,1705,1706,1707,1708,1709,1710,1711,1712,1713,1714,1715,1716,1717,1718,1719,1720,1721,1722,1723,1724,1725,1726,1727,1728,1729,1730,1731,1732,1733,1734,1735,1736,1737,1738,1739,1740,1741,1742,1743,1744,1745,1746,1747,1748,1749,1750,1751,1752,1753,1754,1755,1756,1757,1758,1759,1760,1761,1762,1763,1764,1765,1766,1767,1768,1769,1770,1771,1772,1773,1774,1775,1776,1777,1778,1779,1780,1781,1782,1783,1784,1785,1786,1787,1788,1789,1790,1791,1792,1793,1794,1795,1796,1797,1798,1799,1800,1801,1802,1803,1804,1805,1806,1807,1808,1809,1810,1811,1812,1813,1814,1815,1816,1817,1818,1819,1820,1821,1822,1823,1824,1825,1826,1827,1828,1829,1830,1831,1832,1833,1834,1835,1836,1837,1838,1839,1840,1841,1842,1843,1844,1845,1846,1847,1848,1849,1850,1851,1852,1853,1854,1855,1856,1857,1858,1859,1860,1861,1862,1863,1864,1865,1866,1867,1868,1869,1870,1871,1872,1873,1874,1875,1876,1877,1878,1879,1880,1881,1882,1883,1884,1885,1886,1887,1888,1889,1890,1891,1892,1893,1894,1895,1896,1897,1898,1899,1900,1901,1902,1903,1904,1905,1906,1907,1908,1909,1910,1911,1912,1913,1914,1915,1916,1917,1918,1919,1920,1921,1922,1923,1924,1925,1926,1927,1928,1929,1930,1931,1932,1933,1934,1935,1936,1937,1938,1939,1940,1941,1942,1943,1944,1945,1946,1947,1948,1949,1950,1951,1952,1953,1954,1955,1956,1957,1958,1959,1960,1961,1962,1963,1964,1965,1966,1967,1968,1969,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035,2036,2037,2038,2039,2040,2041,2042,2043,2044,2045,2046,2047,2048,2049,2050,2051,2052,2053,2054,2055,2056,2057,2058,2059,2060,2061,2062,2063,2064,2065,2066,2067,2068,2069,2070,2071,2072,2073,2074,2075,2076,2077,2078,2079,2080,2081,2082,2083,2084,2085,2086,2087,2088,2089,2090,2091,2092,2093,2094,2095,2096,2097,2098,2099,2100,2101,2102,2103,2104,2105,2106,2107,2108,2109,2110,2111,2112,2113,2114,2115,2116,2117,2118,2119,2120,2121,2122,2123,2124,2125,2126,2127,2128,2129,2130,2131,2132,2133,2134,2135,2136,2137,2138,2139,2140,2141,2142,2143,2144,2145,2146,2147,2148,2149,2150,2151,2152,2153,2154,2155,2156,2157,2158,2159,2160,2161,2162,2163,2164,2165,2166,2167,2168,2169,2170,2171,2172,2173,2174,2175,2176,2177,2178,2179,2180,2181,2182,2183,2184,2185,2186,2187,2188,2189,2190,2191,2192,2193,2194,2195,2196,2197,2198,2199,2200,2201,2202,2203,2204,2205,2206,2207,2208,2209,2210,2211,2212,2213,2214,2215,2216,2217,2218,2219,2220,2221,2222,2223,2224,2225,2226,2227,2228,2229,2230,2231,2232,2233,2234,2235,2236,2237,2238,2239,2240,2241,2242,2243,2244,2245,2246,2247,2248,2249,2250,2251,2252,2253,2254,2255,2256,2257,2258,2259,2260,2261,2262,2263,2264,2265,2266,2267,2268,2269,2270,2271,2272,2273,2274,2275,2276,2277,2278,2279,2280,2281,2282,2283,2284,2285,2286,2287,2288,2289,2290,2291,2292,2293,2294,2295,2296,2297,2298,2299,2300,2301,2302,2303,2304,2305,2306,2307,2308,2309,2310,2311,2312,2313,2314,2315,2316,2317,2318,2319,2320,2321,2322,2323,2324,2325,2326,2327,2328,2329,2330,2331,2332,2333,2334,2335,2336,2337,2338,2339,2340,2341,2342,2343,2344,2345,2346,2347,2348,2349,2350,2351,2352,2353,2354,2355,2356,2357,2358,2359,2360,2361,2362,2363,2364,2365,2366,2367,2368,2369,2370,2371,2372,2373,2374,2375,2376,2377,2378,2379,2380,2381,2382,2383,2384,2385,2386,2387,2388,2389,2390,2391,2392,2393,2394,2395,2396,2397,2398,2399,2400,2401,2402,2403,2404,2405,2406,2407,2408,2409,2410,2411,2412,2413,2414,2415,2416,2417,2418,2419,2420,2421,2422,2423,2424,2425,2426,2427,2428,2429,2430,2431,2432,2433,2434,2435,2436,2437,2438,2439,2440,2441,2442,2443,2444,2445,2446,2447,2448,2449,2450,2451,2452,2453,2454,2455,2456,2457,2458,2459,2460,2461,2462,2463,2464,2465,2466,2467,2468,2469,2470,2471,2472,2473,2474,2475,2476,2477,2478,2479,2480,2481,2482,2483,2484,2485,2486,2487,2488,2489,2490,2491,2492,2493,2494,2495,2496,2497,2498,2499,2500,2501,2502,2503,2504,2505,2506,2507,2508,2509,2510,2511,2512,2513,2514,2515,2516,2517,2518,2519,2520,2521,2522,2523,2524,2525,2526,2527,2528,2529,2530,2531,2532,2533,2534,2535,2536,2537,2538,2539,2540,2541,2542,2543,2544,2545,2546,2547,2548,2549,2550,2551,2552,2553,2554,2555,2556,2557,2558,2559,2560,2561,2562,2563,2564,2565,2566,2567,2568,2569,2570,2571,2572,2573,2574,2575,2576,2577,2578,2579,2580,2581,2582,2583,2584,2585,2586,2587,2588,2589,2590,2591,2592,2593,2594,2595,2596,2597,2598,2599,2600,2601,2602,2603,2604,2605,2606,2607,2608,2609,2610,2611,2612,2613,2614,2615,2616,2617,2618,2619,2620,2621,2622,2623,2624,2625,2626,2627,2628,2629,2630,2631,2632,2633,2634,2635,2636,2637,2638,2639,2640,2641,2642,2643,2644,2645,2646,2647,2648,2649,2650,2651,2652,2653,2654,2655,2656,2657,2658,2659,2660,2661,2662,2663,2664,2665,2666,2667,2668,2669,2670,2671,2672,2673,2674,2675,2676,2677,2678,2679,2680,2681,2682,2683,2684,2685,2686,2687,2688,2689,2690,2691,2692,2693,2694,2695,2696,2697,2698,2699,2700,2701,2702,2703,2704,2705,2706,2707,2708,2709,2710,2711,2712,2713,2714,2715,2716,2717,2718,2719,2720,2721,2722,2723,2724,2725,2726,2727,2728,2729,2730,2731,2732,2733,2734,2735,2736,2737,2738,2739,2740,2741,2742,2743,2744,2745,2746,2747,2748,2749,2750,2751,2752,2753,2754,2755,2756,2757,2758,2759,2760,2761,2762,2763,2764,2765,2766,2767,2768,2769,2770,2771,2772,2773,2774,2775,2776,2777,2778,2779,2780,2781,2782,2783,2784,2785,2786,2787,2788,2789,2790,2791,2792,2793,2794,2795,2796,2797,2798,2799,2800,2801,2802,2803,2804,2805,2806,2807,2808,2809,2810,2811,2812,2813,2814,2815,2816,2817,2818,2819,2820,2821,2822,2823,2824,2825,2826,2827,2828,2829,2830,2831,2832,2833,2834,2835,2836,2837,2838,2839,2840,2841,2842,2843,2844,2845,2846,2847,2848,2849,2850,2851,2852,2853,2854,2855,2856,2857,2858,2859,2860,2861,2862,2863,2864,2865,2866,2867,2868,2869,2870,2871,2872,2873,2874,2875,2876,2877,2878,2879,2880,2881,2882,2883,2884,2885,2886,2887,2888,2889,2890,2891,2892,2893,2894,2895,2896,2897,2898,2899,2900,2901,2902,2903,2904,2905,2906,2907,2908,2909,2910,2911,2912,2913,2914,2915,2916,2917,2918,2919,2920,2921,2922,2923,2924,2925,2926,2927,2928,2929,2930,2931,2932,2933,2934,2935,2936,2937,2938,2939,2940,2941,2942,2943,2944,2945,2946,2947,2948,2949,2950,2951,2952,2953,2954,2955,2956,2957,2958,2959,2960,2961,2962,2963,2964,2965,2966,2967,2968,2969,2970,2971,2972,2973,2974,2975,2976,2977,2978,2979,2980,2981,2982,2983,2984,2985,2986,2987,2988,2989,2990,2991,2992,2993,2994,2995,2996,2997,2998,2999,3000,3001,3002,3003,3004,3005,3006,3007,3008,3009,3010,3011,3012,3013,3014,3015,3016,3017,3018,3019,3020,3021,3022,3023,3024,3025,3026,3027,3028,3029,3030,3031,3032,3033,3034,3035,3036,3037,3038,3039,3040,3041,3042,3043,3044,3045,3046,3047,3048,3049,3050,3051,3052,3053,3054,3055,3056,3057,3058,3059,3060,3061,3062,3063,3064,3065,3066,3067,3068,3069,3070,3071,3072,3073,3074,3075,3076,3077,3078,3079,3080,3081,3082,3083,3084,3085,3086,3087,3088,3089,3090,3091,3092,3093,3094,3095,3096,3097,3098,3099,3100,3101,3102,3103,3104,3105,3106,3107,3108,3109,3110,3111,3112,3113,3114,3115,3116,3117,3118,3119,3120,3121,3122,3123,3124,3125,3126,3127,3128,3129,3130,3131,3132,3133,3134,3135,3136,3137,3138,3139,3140,3141,3142,3143,3144,3145,3146,3147,3148,3149,3150,3151,3152,3153,3154,3155,3156,3157,3158,3159,3160,3161,3162,3163,3164,3165,3166,3167,3168,3169,3170,3171,3172,3173,3174,3175,3176,3177,3178,3179,3180,3181,3182,3183,3184,3185,3186,3187,3188,3189,3190,3191,3192,3193,3194,3195,3196,3197,3198,3199,3200,3201,3202,3203,3204,3205,3206,3207,3208,3209,3210,3211,3212,3213,3214,3215,3216,3217,3218,3219,3220,3221,3222,3223,3224,3225,3226,3227,3228,3229,3230,3231,3232,3233,3234,3235,3236,3237,3238,3239,3240,3241,3242,3243,3244,3245,3246,3247,3248,3249,3250,3251,3252,3253,3254,3255,3256,3257,3258,3259,3260,3261,3262,3263,3264,3265,3266,3267,3268,3269,3270,3271,3272,3273,3274,3275,3276,3277,3278,3279,3280,3281,3282,3283,3284,3285,3286,3287,3288,3289,3290,3291,3292,3293,3294,3295,3296,3297,3298,3299,3300,3301,3302,3303,3304,3305,3306,3307,3308,3309,3310,3311,3312,3313,3314,3315,3316,3317,3318,3319,3320,3321,3322,3323,3324,3325,3326,3327,3328,3329,3330,3331,3332,3333,3334,3335,3336,3337,3338,3339,3340,3341,3342,3343,3344,3345,3346,3347,3348,3349,3350,3351,3352,3353,3354,3355,3356,3357,3358,3359,3360,3361,3362,3363,3364,3365,3366,3367,3368,3369,3370,3371,3372,3373,3374,3375,3376,3377,3378,3379,3380,3381,3382,3383,3384,3385,3386,3387,3388,3389,3390,3391,3392,3393,3394,3395,3396,3397,3398,3399,3400,3401,3402,3403,3404,3405,3406,3407,3408,3409,3410,3411,3412,3413,3414,3415,3416,3417,3418,3419,3420,3421,3422,3423,3424,3425,3426,3427,3428,3429,3430,3431,3432,3433,3434,3435,3436,3437,3438,3439,3440,3441,3442,3443,3444,3445,3446,3447,3448,3449,3450,3451,3452,3453,3454,3455,3456,3457,3458,3459,3460,3461,3462,3463,3464,3465,3466,3467,3468,3469,3470,3471,3472,3473,3474,3475,3476,3477,3478,3479,3480,3481,3482,3483,3484,3485,3486,3487,3488,3489,3490,3491,3492,3493,3494,3495,3496,3497,3498,3499,3500,3501,3502,3503,3504,3505,3506,3507,3508,3509,3510,3511,3512,3513,3514,3515,3516,3517,3518,3519,3520,3521,3522,3523,3524,3525,3526,3527,3528,3529,3530,3531,3532,3533,3534,3535,3536,3537,3538,3539,3540,3541,3542,3543,3544,3545,3546,3547,3548,3549,3550,3551,3552,3553,3554,3555,3556,3557,3558,3559,3560,3561,3562,3563,3564,3565,3566,3567,3568,3569,3570,3571,3572,3573,3574,3575,3576,3577,3578,3579,3580,3581,3582,3583,3584,3585,3586,3587,3588,3589,3590,3591,3592,3593,3594,3595,3596,3597,3598,3599,3600,3601,3602,3603,3604,3605,3606,3607,3608,3609,3610,3611,3612,3613,3614,3615,3616,3617,3618,3619,3620,3621,3622,3623,3624,3625,3626,3627,3628,3629,3630,3631,3632,3633,3634,3635,3636,3637,3638,3639,3640,3641,3642,3643,3644,3645,3646,3647,3648,3649,3650,3651,3652,3653,3654,3655,3656,3657,3658,3659,3660,3661,3662,3663,3664,3665,3666,3667,3668,3669,3670,3671,3672,3673,3674,3675,3676,3677,3678,3679,3680,3681,3682,3683,3684,3685,3686,3687,3688,3689,3690,3691,3692,3693,3694,3695,3696,3697,3698,3699,3700,3701,3702,3703,3704,3705,3706,3707,3708,3709,3710,3711,3712,3713,3714,3715,3716,3717,3718,3719,3720,3721,3722,3723,3724,3725,3726,3727,3728,3729,3730,3731,3732,3733,3734,3735,3736,3737,3738,3739,3740,3741,3742,3743,3744,3745,3746,3747,3748,3749,3750,3751,3752,3753,3754,3755,3756,3757,3758,3759,3760,3761,3762,3763,3764,3765,3766,3767,3768,3769,3770,3771,3772,3773,3774,3775,3776,3777,3778,3779,3780,3781,3782,3783,3784,3785,3786,3787,3788,3789,3790,3791,3792,3793,3794,3795,3796,3797,3798,3799,3800,3801,3802,3803,3804,3805,3806,3807,3808,3809,3810,3811,3812,3813,3814,3815,3816,3817,3818,3819,3820,3821,3822,3823,3824,3825,3826,3827,3828,3829,3830,3831,3832,3833,3834,3835,3836,3837,3838,3839,3840,3841,3842,3843,3844,3845,3846,3847,3848,3849,3850,3851,3852,3853,3854,3855,3856,3857,3858,3859,3860,3861,3862,3863,3864,3865,3866,3867,3868,3869,3870,3871,3872,3873,3874,3875,3876,3877,3878,3879,3880,3881,3882,3883,3884,3885,3886,3887,3888,3889,3890,3891,3892,3893,3894,3895,3896,3897,3898,3899,3900,3901,3902,3903,3904,3905,3906,3907,3908,3909,3910,3911,3912,3913,3914,3915,3916,3917,3918,3919,3920,3921,3922,3923,3924,3925,3926,3927,3928,3929,3930,3931,3932,3933,3934,3935,3936,3937,3938,3939,3940,3941,3942,3943,3944,3945,3946,3947,3948,3949,3950,3951,3952,3953,3954,3955,3956,3957,3958,3959,3960,3961,3962,3963,3964,3965,3966,3967,3968,3969,3970,3971,3972,3973,3974,3975,3976,3977,3978,3979,3980,3981,3982,3983,3984,3985,3986,3987,3988,3989,3990,3991,3992,3993,3994,3995,3996,3997,3998,3999,4000,4001,4002,4003,4004,4005,4006,4007,4008,4009,4010,4011,4012,4013,4014,4015,4016,4017,4018,4019,4020,4021,4022,4023,4024,4025,4026,4027,4028,4029,4030,4031,4032,4033,4034,4035,4036,4037,4038,4039,4040,4041,4042,4043,4044,4045,4046,4047,4048,4049,4050,4051,4052,4053,4054,4055,4056,4057,4058,4059,4060,4061,4062,4063,4064,4065,4066,4067,4068,4069,4070,4071,4072,4073,4074,4075,4076,4077,4078,4079,4080,4081,4082,4083,4084,4085,4086,4087,4088,4089,4090,4091,4092,4093,4094,4095,4096,4097,4098,4099,4100,4101,4102,4103,4104,4105,4106,4107,4108,4109,4110,4111,4112,4113,4114,4115,4116,4117,4118,4119,4120,4121,4122,4123,4124,4125,4126,4127,4128,4129,4130,4131,4132,4133,4134,4135,4136,4137,4138,4139,4140,4141,4142,4143,4144,4145,4146,4147,4148,4149,4150,4151,4152,4153,4154,4155,4156,4157,4158,4159,4160,4161,4162,4163,4164,4165,4166,4167,4168,4169,4170,4171,4172,4173,4174,4175,4176,4177,4178,4179,4180,4181,4182,4183,4184,4185,4186,4187,4188,4189,4190,4191,4192,4193,4194,4195,4196,4197,4198,4199,4200,4201,4202,4203,4204,4205,4206,4207,4208,4209,4210,4211,4212,4213,4214,4215,4216,4217,4218,4219,4220,4221,4222,4223,4224,4225,4226,4227,4228,4229,4230,4231,4232,4233,4234,4235,4236,4237,4238,4239,4240,4241,4242,4243,4244,4245,4246,4247,4248,4249,4250,4251,4252,4253,4254,4255,4256,4257,4258,4259,4260,4261,4262,4263,4264,4265,4266,4267,4268,4269,4270,4271,4272,4273,4274,4275,4276,4277,4278,4279,4280,4281,4282,4283,4284,4285,4286,4287,4288,4289,4290,4291,4292,4293,4294,4295,4296,4297,4298,4299,4300,4301,4302,4303,4304,4305,4306,4307,4308,4309,4310,4311,4312,4313,4314,4315,4316,4317,4318,4319,4320,4321,4322,4323,4324,4325,4326,4327,4328,4329,4330,4331,4332,4333,4334,4335,4336,4337,4338,4339,4340,4341,4342,4343,4344,4345,4346,4347,4348,4349,4350,4351,4352,4353,4354,4355,4356,4357,4358,4359,4360,4361,4362,4363,4364,4365,4366,4367,4368,4369,4370,4371,4372,4373,4374,4375,4376,4377,4378,4379,4380,4381,4382,4383,4384,4385,4386,4387,4388,4389,4390,4391,4392,4393,4394,4395,4396,4397,4398,4399,4400,4401,4402,4403,4404,4405,4406,4407,4408,4409,4410,4411,4412,4413,4414,4415,4416,4417,4418,4419,4420,4421,4422,4423,4424,4425,4426,4427,4428,4429,4430,4431,4432,4433,4434,4435,4436,4437,4438,4439,4440,4441,4442,4443,4444,4445,4446,4447,4448,4449,4450,4451,4452,4453,4454,4455,4456,4457,4458,4459,4460,4461,4462,4463,4464,4465,4466,4467,4468,4469,4470,4471,4472,4473,4474,4475,4476,4477,4478,4479,4480,4481,4482,4483,4484,4485,4486,4487,4488,4489,4490,4491,4492,4493,4494,4495,4496,4497,4498,4499,4500,4501,4502,4503,4504,4505,4506,4507,4508,4509,4510,4511,4512,4513,4514,4515,4516,4517,4518,4519,4520,4521,4522,4523,4524,4525,4526,4527,4528,4529,4530,4531,4532,4533,4534,4535,4536,4537,4538,4539,4540,4541,4542,4543,4544,4545,4546,4547,4548,4549,4550,4551,4552,4553,4554,4555,4556,4557,4558,4559,4560,4561,4562,4563,4564,4565,4566,4567,4568,4569,4570,4571,4572,4573,4574,4575,4576,4577,4578,4579,4580,4581,4582,4583,4584,4585,4586,4587,4588,4589,4590,4591,4592,4593,4594,4595,4596,4597,4598,4599,4600,4601,4602,4603,4604,4605,4606,4607,4608,4609,4610,4611,4612,4613,4614,4615,4616,4617,4618,4619,4620,4621,4622,4623,4624,4625,4626,4627,4628,4629,4630,4631,4632,4633,4634,4635,4636,4637,4638,4639,4640,4641,4642,4643,4644,4645,4646,4647,4648,4649,4650,4651,4652,4653,4654,4655,4656,4657,4658,4659,4660,4661,4662,4663,4664,4665,4666,4667,4668,4669,4670,4671,4672,4673,4674,4675,4676,4677,4678,4679,4680,4681,4682,4683,4684,4685,4686,4687,4688,4689,4690,4691,4692,4693,4694,4695,4696,4697,4698,4699,4700,4701,4702,4703,4704,4705,4706,4707,4708,4709,4710,4711,4712,4713,4714,4715,4716,4717,4718,4719,4720,4721,4722,4723,4724,4725,4726,4727,4728,4729,4730,4731,4732,4733,4734,4735,4736,4737,4738,4739,4740,4741,4742,4743,4744,4745,4746,4747,4748,4749,4750,4751,4752,4753,4754,4755,4756,4757,4758,4759,4760,4761,4762,4763,4764,4765,4766,4767,4768,4769,4770,4771,4772,4773,4774,4775,4776,4777,4778,4779,4780,4781,4782,4783,4784,4785,4786,4787,4788,4789,4790,4791,4792,4793,4794,4795,4796,4797,4798,4799,4800,4801,4802,4803,4804,4805,4806,4807,4808,4809,4810,4811,4812,4813,4814,4815,4816,4817,4818,4819,4820,4821,4822,4823,4824,4825,4826,4827,4828,4829,4830,4831,4832,4833,4834,4835,4836,4837,4838,4839,4840,4841,4842,4843,4844,4845,4846,4847,4848,4849,4850,4851,4852,4853,4854,4855,4856,4857,4858,4859,4860,4861,4862,4863,4864,4865,4866,4867,4868,4869,4870,4871,4872,4873,4874,4875,4876,4877,4878,4879,4880,4881,4882,4883,4884,4885,4886,4887,4888,4889,4890,4891,4892,4893,4894,4895,4896,4897,4898,4899,4900,4901,4902,4903,4904,4905,4906,4907,4908,4909,4910,4911,4912,4913,4914,4915,4916,4917,4918,4919,4920,4921,4922,4923,4924,4925,4926,4927,4928,4929,4930,4931,4932,4933,4934,4935,4936,4937,4938,4939,4940,4941,4942,4943,4944,4945,4946,4947,4948,4949,4950,4951,4952,4953,4954,4955,4956,4957,4958,4959,4960,4961,4962,4963,4964,4965,4966,4967,4968,4969,4970,4971,4972,4973,4974,4975,4976,4977,4978,4979,4980,4981,4982,4983,4984,4985,4986,4987,4988,4989,4990,4991,4992,4993,4994,4995,4996,4997,4998,4999])               
import matplotlib.pyplot as plt

plt.plot(t)        
        
a=[1,2,3,4,5,6]        
b=a[:]       

a[0]=9
print(a,b) 

#328. Odd Even Linked List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        
        odd=head
        even=head.next
        evenhead=head.next
        
        
        while even and even.next:
            odd.next=odd.next.next
            odd=odd.next
            even.next=even.next.next
            even=even.next
        odd.next=evenhead
        return head
#331. Verify Preorder Serialization of a Binary Tree        
class Solution:
    def isValidSerialization(self, preorder):
        """
        :type preorder: str
        :rtype: bool
        """
        p=  preorder.split(',')
        
        slot=1
        
        for node in p:
            
            if slot==0:
                return False
            
            if node=='#':
               slot-=1
               
            else:
                slot+=1
        return slot==0
                
#332. Reconstruct Itinerary        
class Solution:
    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        from collections import defaultdict , deque
        
        def buildG(tickets):
            G=defaultdict(list)
            
            for S,E in tickets:
                G[S].append(E)
                
            for A in G:
                G[A].sort(reverse=True)
                G[A]=deque(G[A])
            return G
        
        def dfs(G,S):
            trip.append(S)
            if len(trip)==length:
                return True
            if S in G:
                queue=G[S]
                for _ in range(len(queue)):
                    A=queue.pop()
                    if dfs(G,A):
                        return True
                    
                    queue.appendleft(A)
            trip.pop()
            return False  
        
        
        G=buildG(tickets)
        trip=[]
        length=len(tickets)+1
        dfs(G,'JFK')
        return trip
        
if __name__ == "__main__":
    print(Solution().findItinerary(tickets))                   
        
        
tickets=        [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]       
tickets=[["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]  

#334. Increasing Triplet Subsequence
class Solution:
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """  
        first=second=float('inf')
        
        for num in nums:
            print(first,second)
            if num<=first:
                first=num
            elif num<=second:
                second=num
            else:
                return True
        return False
if __name__ == "__main__":
    print(Solution().increasingTriplet([1,2,3,4,5]))         

#37. House Robber III        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        def subrob(root):
                 
            if not root:
               return [0,0]    
       
            val=0

          
            left=subrob(root.left)
            
        
            right=subrob(root.right)
               
            res=[0,0]
            
            res[0]=max(left[0],left[1])+max(right[0],right[1])
            res[1]=root.val+    left[0]+right[0]
            
            return res
        
        res=subrob(root)
        return max(res[0],res[1])
        
#338. Counting Bits        
class Solution:
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """        
        if not num:
            return 0
        
        res=[]
        for n in range(num+1):
            res.append(bin(n).count('1'))
        return res
if __name__ == "__main__":
    print(Solution().countBits(5))         
            
#341. Flatten Nested List Iterator        
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger(object):
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class NestedIterator(object):
     
    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        import collections
        self.queue = collections.deque([])
        for elem in nestedList:
            if elem.isInteger():
                self.queue.append(elem.getInteger())
            else:
                newList = NestedIterator(elem.getList())
                while newList.hasNext():
                    self.queue.append(newList.next())

    def next(self):
        """
        :rtype: int
        """
        return self.queue.popleft()

    def hasNext(self):
        """
        :rtype: bool
        """
        if self.queue:
            return True
        return False
    
t=NestedIterator([[1,1],2,[1,1]])
# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())        
        
#343. Integer Break        
class Solution:
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """

        res=[0 for _ in range(n+1)]  
        
        for i in range(5):
            res[i]=i
        
        for i in range(2,n+1):
            for j in range(i):
                res[i]=max(res[i],res[i-j]*j)
        return res[-1]
        
#347. Top K Frequent Elements        
class Solution:
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        from    collections import defaultdict  
        d=defaultdict(int)
        
        for num in nums:
            d[num]+=1
        
        v=[]
        
        for key ,value in d.items():
            v+=[value]
        
        v.sort(reverse=True)
        
        res=[]
        for key ,value in d.items():
            if value in v[:k]:
                res.append(key)
        return res
if __name__ == "__main__":
    print(Solution().topKFrequent([1,1,1,2,2,3],2))              
        
#355. Design Twitter        
class Twitter:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        import collections
        import itertools
        self.timer=itertools.count(step=-1)
        self.tweets=collections.defaultdict(collections.deque) 
        self.followees=collections.defaultdict(set) 
        
        
    def postTweet(self, userId, tweetId):
        """
        Compose a new tweet.
        :type userId: int
        :type tweetId: int
        :rtype: void
        """
        self.tweets[userId].appendleft((next(self.timer),tweetId))

    def getNewsFeed(self, userId):
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        :type userId: int
        :rtype: List[int]
        """
        import heapq
        import itertools
        tweet=heapq.merge(*(self.tweets[u] for u in self.followees[userId] | {userId} ))
        return [t for _, t in itertools.islice(tweet,10)]

    def follow(self, followerId, followeeId):
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        self.followees[followerId].add(followeeId)

    def unfollow(self, followerId, followeeId):
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        self.followees[followerId].discard(followeeId)


# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)        
        
#357. Count Numbers with Unique Digits        
class Solution:
    def countNumbersWithUniqueDigits(self, n):
        """
        :type n: int
        :rtype: int
        """

        choice=[9,9,8,7,6,5,4,3,2,1] 

        ans=1
        product=1
       
        for i in range(n if n<10 else 10):
           product*=choice[i]
           ans+=product
        
        return ans
        
if __name__ == "__main__":
    print(Solution().countNumbersWithUniqueDigits(0))             
        
#365. Water and Jug Problem        
class Solution:
    def canMeasureWater(self, x, y, z):
        """
        :type x: int
        :type y: int
        :type z: int
        :rtype: bool
        """     
        from math import gcd
        
        if x+y<z:
            return False
        
        if x==y or y==z or x+y==z:
            return True
        
        if z%gcd(x,y)==0:
            return True
        return False
        
        
        return search(values,z)
if __name__ == "__main__":
    print(Solution().canMeasureWater(34,5,6))         
        
#368. Largest Divisible Subset        
class Solution:
    def largestDivisibleSubset(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """        
        n=len(nums)
        if n==0:
            return []
        from copy import copy
        dp=[0 for _ in range(n)]
        nums.sort()
        dp[0]=[nums[0]]
        
        for i in range(1,n):
            maxsize=0
            maxset=[]
            curnum=nums[i]
            for j in range(i):
                
                if curnum%nums[j]==0:
                   print(dp[j])
                   localset=list(dp[j])
                   if len(localset) >maxsize:
                      maxset=list(localset)
                      maxsize=len(maxset)
                      print(maxset)
            
            maxset.append(curnum)
            dp[i]=list(maxset)
            #print(dp)
        finalsize=0
        res=[]
        for localset  in dp:
            if len(localset)>finalsize:
                res=localset
                finalsize=len(localset)
        return res
nums=[1,2,4,8]
if __name__ == "__main__":
    print(Solution().largestDivisibleSubset([1,2,4,8]))          
        
#372. Super Pow        
class Solution:
    def superPow(self, a, b):
        """
        :type a: int
        :type b: List[int]
        :rtype: int
        """
        result=1
        
        
        
        
        for digit in  b:
            result=pow(result,10,1337)*pow(a,digit,1337)%1337
        return result

#373. Find K Pairs with Smallest Sums
class Solution:
    def kSmallestPairs(self, nums1, nums2, k):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type k: int
        :rtype: List[List[int]]
        """
        
        from collections import defaultdict
        n1=len(nums1)
        n2=len(nums2)
        
        if n1==0 or n2==0:
            return []
        
        if n1*n2<=k:
            return [[a,b] for a in nums1 for b in nums2]
        
        d1=[]
        d2=defaultdict(list)
        allcombine=[[a,b] for a in nums1 for b in nums2]
        for i in range(len(allcombine)):
            d1.append(sum(allcombine[i]))
            d2[sum(allcombine[i])].append(allcombine[i])
        
        
        d1.sort()
        print(d1)
        res=[]
        
        for i in range(k):
            if i>0 and d1[i]==d1[i-1]:
                continue
         
            
            res+=d2[d1[i]]
        
        return res[:k]
nums1= [0,0,0,0,0]
nums2=[-3,22,35,56,76]
if __name__ == "__main__":
    print(Solution().kSmallestPairs(nums1, nums2, 22))               
            
#375. Guess Number Higher or Lower II  
class Solution:
    def getMoneyAmount(self, n):
        """
        :type n: int
        :rtype: int
        """
       
        
        
        dp=[[0] * (n+1) for _ in range(n+1)]
       
        for lo in range(n,0,-1):
            for hi in range(lo+1,n+1):
                temp=float('inf')#print(lo,hi)
                for x in range(lo,hi):
                     
                     
                     #print(temp)
                     print(x+dp[lo][x-1],x+dp[x+1][hi])
                     if temp>x+max(dp[lo][x-1],dp[x+1][hi]):
                        temp=x+max(dp[lo][x-1],dp[x+1][hi])
                dp[lo][hi]=temp
        #print(dp)
        return dp[1][n]

class Solution:
 def getMoneyAmount(self, n):
    need = [[0] * (n+1) for _ in range(n+1)]
    for lo in range(n, 0, -1):
        for hi in range(lo+1, n+1):
            need[lo][hi] = min(x + max(need[lo][x-1], need[x+1][hi])
                               for x in range(lo, hi))
    return need[1][n]    
    
if __name__ == "__main__":
    print(Solution().getMoneyAmount(6))            
                
#376. Wiggle Subsequence            
class Solution:
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n=len(nums)
        if n<2:
            return n
        
        up=[0 for _ in range(n)]
        down=[0 for _ in range(n)]
        
        up[0]=1
        down[0]=1
        
        for i in range(1,n):
            if nums[i]>nums[i-1]:
               up[i]=down[i-1]+1
               down[i]=down[i-1]
            elif nums[i]<nums[i-1]:
               up[i]=up[i-1]
               down[i]=up[i-1]+1
            else:
               up[i]=up[i-1]
               down[i]=down[i-1]
        return max(down[n-1],up[n-1])
if __name__ == "__main__":
    print(Solution().wiggleMaxLength( [1,17,5,10,13,15,10,5,16,8]))            
                
#377. Combination Sum IV
class Solution:
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if not nums:
            return 0
        dp=[0 for _ in range(target+1)]
        dp[0]=1
        
        for i in range(1,target+1):
            for j in range(len(nums)):
                if i-nums[j]>=0:
                    dp[i]+=dp[i-nums[j]]
        return dp[target]
if __name__ == "__main__":
    print(Solution().combinationSum4( [4,2,1],32))                      
        
              
#378. Kth Smallest Element in a Sorted Matrix        
class Solution:
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        if len(matrix)==0:
            return 0
        if len(matrix[0])==0:
            return 0
        
        from heapq import heappush,heapify,heappop
        
        heap=[(row[0],i,0) for i,row in enumerate(matrix)]
        
        heapify(heap)
        
        for _ in range(k):
            item,i,j=heappop(heap)
            
            if j+1<len(matrix[0]):
               heappush(heap,(matrix[i][j+1],i,j+1))
        return item
        
        
        
        
matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
]
k = 8
if __name__ == "__main__":
    print(Solution().kthSmallest(matrix, k))                  
#380. Insert Delete GetRandom O(1)
class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.nums=[]
        self.pos={}

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.pos:
           self.nums.append(val)
           self.pos[val]=len(self.nums)-1
           return True
        return False
            

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.pos:
            idx=self.pos[val]
            last=self.nums[-1]
            self.nums[idx]=last
            self.pos[last]=idx
            self.pos.pop(val,0)
            self.nums.pop()
            return True
        return False
            

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        import random
        return self.nums[random.randint(0,len(self.nums)-1)]
        
        


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
#382. Linked List Random Node
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:

    def __init__(self, head):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        :type head: ListNode
        """
        self.head=head

    def getRandom(self):
        """
        Returns a random node's value.
        :rtype: int
        """
        import random
        result=self.head
        node=self.head.next
        index=1
        while node:
            if random.randint(0,index)==0:
                result=node
            node=node.next
            index+=1
        return result.val
            


# Your Solution object will be instantiated and called as such:
# obj = Solution(head)
# param_1 = obj.getRandom()

#384. Shuffle an Array
class Solution:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.array=nums
        self.original=list(nums)

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        :rtype: List[int]
        """
        self.array=self.original
        self.original=list(self.original)
        return self.array
        

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        :rtype: List[int]
        """
        import random
        for idx in range(len(self.array)):
            randomidx=random.randrange(idx,len(self.array))
            self.array[idx],self.array[randomidx]=self.array[randomidx],self.array[idx]
        return self.array
        


# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.reset()
# param_2 = obj.shuffle()        

#385. Mini Parser
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger:
#    def __init__(self, value=None):
#        """
#        If value is not specified, initializes an empty list.
#        Otherwise initializes a single integer equal to value.
#        """
#
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def add(self, elem):
#        """
#        Set this NestedInteger to hold a nested list and adds a nested integer elem to it.
#        :rtype void
#        """
#
#    def setInteger(self, value):
#        """
#        Set this NestedInteger to hold a single integer equal to value.
#        :rtype void
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class Solution:
    def deserialize(self, s):
        """
        :type s: str
        :rtype: NestedInteger
        """
        if s[0]!='[':
            return NestedInteger(int(s))
        numP=0
        start=1
        
        nest=NestedInteger()
        for i in range(1,len(s)):
            if (numP==0 and s[i]==',')  or i==len(s)-1:
                if start<i:
                    
                   nest.add(self.deserialize(s[start:i]))
                start=i+1
            elif s[i]=='[':
                 numP+=1
            elif s[i]==']':
                 numP-=1
        return nest
#386. Lexicographical Numbers        
class Solution:
    def lexicalOrder(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
            
        string=[str(i) for i in range(1,n+1)]
        string.sort()
        nums=[int(i) for i in string]
        return nums
if __name__ == "__main__":
    print(Solution().lexicalOrder(13)) 
        
#388. Longest Absolute File Path        
class Solution:
    def lengthLongestPath(self, input):
        """
        :type input: str
        :rtype: int
        """

        maxlen=0
        pathlen={0:0}
        

        for line in input.splitlines():
            print(line)
            name=line.lstrip('\t')
            depth=len(line)-len(name)
            if '.' in name:
                
                maxlen=max(maxlen,pathlen[depth]+len(name))
            else:
                
                pathlen[depth+1]=pathlen[depth]+len(name)+1
        return maxlen
s='dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext'
if __name__ == "__main__":
    print(Solution().lengthLongestPath(s))         
        
#390. Elimination Game
class Solution:
    def lastRemaining(self, n):
        """
        :type n: int
        :rtype: int
        """
        head=1
        step=1
        left=True
        remaining=n
        
        while remaining>1:
            
            if left  or remaining%2==1:
                head=head+step
            left=  not left
            remaining=remaining//2
            
            step=step*2
        return head
            
if __name__ == "__main__":
    print(Solution().lastRemaining(13))          
        
#392. Is Subsequence       
class Solution:
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if not s:
            return True
        
        i=0
        for j in range(len(t)):
            if t[j]==s[i]:
                i+=1
            if i==len(s):
                return True
        return False

if __name__ == "__main__":
    print(Solution().isSubsequence("abc","ahbgdc")) 

#393. UTF-8 Validation
class Solution:
    def validUtf8(self, data):
        """
        :type data: List[int]
        :rtype: bool
        """
        #http://blog.csdn.net/MebiuW/article/details/52445248
        def countone(num):
            count=0
            for i in range(7,0,-1):
                if num >>i & 1==1:
                    count+=1
                else:
                    break
            return count
        
        ind=0
        n=len(data)
        
        while ind<n:
            
            m=countone(data[ind])
            
            if m+ind>n:
                return False
            elif m==1 or m>4:
                return False
            elif m==0:
                 ind+=1
            else:
                for i in range(ind+1,ind+m):
                    if countone(data[i])!=1:
                        return False
            ind+=m
        return True
            
if __name__ == "__main__":
    print(Solution().validUtf8([197, 130, 1]))             
        
#394. Decode String
class Solution:
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
#        s = "3[a]2[bc]", return "aaabcbc".
#        s = "3[a2[c]]", return "accaccacc".
#        s = "2[abc]3[cd]ef", return "abcabccdcdcdef"        
        
        stack=[]
        curnum=0
        previousstr=''
        curstr=''
        
        for c in s:
            if c =='[':
               stack.append(curstr)
               stack.append(curnum)
               curstr=''
               curnum=0
            elif c ==']':
                 num=stack.pop()
                 previousstr=stack.pop()
                 curstr=previousstr+curstr*num
            elif c.isdigit():
                 curnum=curnum*10+int(c)
            else:
                
                 curstr+=c
        return curstr
if __name__ == "__main__":
    print(Solution().decodeString("2[abc]3[cd]ef"))        


#395. Longest Substring with At Least K Repeating Characters                
class Solution:
    def longestSubstring(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        
        
        for c in set(s):
            if s.count(c)<k:
                return max(self.longestSubstring(t,k) for t in s.split(c))
        return len(s)
if __name__ == "__main__":
    print(Solution().longestSubstring("ababbc",2))        
                      
                
#396. Rotate Function                
class Solution:
    def maxRotateFunction(self, A):
        """
        :type A: List[int]
        :rtype: int
        """                
                
                
#F(k) = F(k-1) + sum - nBk[0]
#What is Bk[0]?
#
#k = 0; B[0] = A[0];
#k = 1; B[0] = A[len-1];
#k = 2; B[0] = A[len-2];     
                
                
        sumN=sum(A)   
        f=0
        
        for i in range(len(A)):
            f+=i*A[i]
        maxf=f
        for i in range(1,len(A)):
            print(f)
            f=f+sumN-len(A)*A[len(A)-i]
            maxf=max(f,maxf)
        return maxf
if __name__ == "__main__":
    print(Solution().maxRotateFunction([4, 3, 2, 6]))    
            
#397. Integer Replacement                
class Solution:
    def integerReplacement(self, n):
        """
        :type n: int
        :rtype: int
        """
                        
# When n is even, the operation is fixed. The procedure is unknown when it is odd. 
# When n is odd it can be written into the form n = 2k+1 (k is a non-negative integer.). 
# That is, n+1 = 2k+2 and n-1 = 2k. Then, (n+1)/2 = k+1 and (n-1)/2 = k. So one of (n+1)/2 
# and (n-1)/2 is even, the other is odd. And the "best" case of this problem is to divide 
# as much as possible. Because of that, always pick n+1 or n-1 based on if it can be divided by 4. 
# The only special case of that is when n=3 you would like to pick n-1 rather than n+1. 
       
        count=0
        while n>1:
            
            if n%2==0:
                n=n/2
            elif (n+1)%4==0 and (n-1)!=2:
                 n+=1
            else:
                n-=1
            count+=1
        return count
if __name__ == "__main__":
    print(Solution().integerReplacement(7))         

#398. Random Pick Index
class Solution:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums=nums
        

    def pick(self, target):
        """
        :type target: int
        :rtype: int
        """
        
        import random
        return random.choice([v for v, f in enumerate(self.nums) if f==target])
        


# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.pick(target)
#399. Evaluate Division
class Solution:
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        graph={}
        
        def buildgraph(equations, values):
            def addedge(f,t,value):
                if f in graph:
                    graph[f].append((t,value))
                else:
                    graph[f]=[(t,value)]
                
                
            for vertice, value in zip(equations, values):
                f,t=vertice
                addedge(f,t,value)
                addedge(t,f,1/value)
            
        
        def find_path(query):
            b,e=query
            if b not in graph or e not in graph:
               return -1
           
            visited=set()
            
            import collections
            q=collections.deque([(b,1)])
            
            while q:
                front, currentproduct=q.popleft()
                if front==e:
                    return currentproduct
                visited.add(front)
                
                for neigbor,value in graph[front]:
                    if neigbor not in visited:
                        q.append((neigbor,currentproduct*value))
            return -1
        
        buildgraph(equations, values)
        
        return [find_path(q)for q in queries]
equations = [ ["a", "b"], ["b", "c"] ]
values = [2.0, 3.0]
queries = [ ["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"] ]
if __name__ == "__main__":
    print(Solution().calcEquation(equations, values, queries)) 

#402. Remove K Digits
class Solution:
    def removeKdigits(self, num, k):
        """
        :type num: str
        :type k: int
        :rtype: str
        """
        
        if k==0:
            return num
        if k==len(num):
            return '0'
        top=0
        stk=[0 for _ in range(len(num))]
        digits=len(num)-k
        for i in range(len(num)):
            c=num[i]
            while top>0  and k>0 and stk[top-1]>c:
                k-=1
                top-=1
            stk[top]=c
            top+=1
        
       
        idx=0
        
       
        while idx<digits  and stk[idx]=='0':
            idx+=1
            print(idx,digits)
        
        if idx==digits:
            return '0'
        else:
            return ''.join(stk[idx:digits])
if __name__ == "__main__":
    print(Solution().removeKdigits("112",1))        
        
#406. Queue Reconstruction by Height            
class Solution:
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """   
        people.sort(key=lambda hk:(-hk[0],hk[1]))
        queue=[]
        for p in people:
            queue.insert(p[1],p)
        return queue
people=[[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]
if __name__ == "__main__":
    print(Solution().reconstructQueue(people)) 
    
#413. Arithmetic Slices    
class Solution:
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
#        
#        1 arithmetic slice of length N (itself);
#2 arithmetic slices of length N-1 (0~N-2 and 1~N-1);
#3 arithmetic slices of length N-2;
#...
#N-2 arithmetic slices of length 3.
#So the total count of arithmetic slices is 1+2+3+...+(N-2) = (N-1)*(N-2)/2.

        curr=0
        sumA=0
        for i in range(2,len(A)):
            if A[i]-A[i-1]==A[i-1]-A[i-2]:
                curr+=1
                sumA+=curr
            else:
                curr=0
        return sumA
if __name__ == "__main__":
    print(Solution().numberOfArithmeticSlices([1, 2, 3, 4]))        
            
           
#416. Partition Equal Subset Sum                
class Solution:
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if not nums:
           return  True

        if sum(nums)%2==1:
           return  False

        target=    sum(nums)/2
        
        dp=[False for _ in range(int(target+1))]
        dp[0]=True
        
        for i in range(len(nums)):
            for j in range(int(target),nums[i]-1,-1):
                dp[j]=dp[j] or dp[j-nums[i]]
        
        
        return dp[int(target)]
nums=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
 ,1,1,1,1,1,1,1,1,1,1,1,1,1,1,100]
if __name__ == "__main__":
    print(Solution().canPartition(nums))                   
            
#417. Pacific Atlantic Water Flow            
class Solution:
    def pacificAtlantic(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        if not matrix:
            return []
        if not matrix[0]:
            return []
        
        m=len(matrix)
        n=len(matrix[0])
        
        p_visited=[[False for _ in range(n)] for _ in range(m)]
        a_visited=[[False for _ in range(n)] for _ in range(m)]
        
        direction=[[0,1], [0,-1], [1,0], [-1,0]]
        def dfs(i,j,visited):
            visited[i][j]=True
            
            for d in direction:
                x,y=i+d[0],  j+d[1]
                if x<0 or x>=m or y<0 or y>=n or  visited[x][y] or matrix[i][j]>matrix[x][y]:
                    continue
                dfs(x,y,visited)
        
        for i in range(m):
             p_visited[i][0]=True
             a_visited[i][n-1]=True
        
        for i in range(n):
             p_visited[0][i]=True
             a_visited[m-1][i]=True
             
        for i in range(m):
             dfs(i,0,p_visited)
             dfs(i,n-1,a_visited)
        
        for i in range(n):
             dfs(0,i,p_visited)
             dfs(m-1,i,a_visited)
        
        res=[]
        
        for i in range(m):
            for j in range(n):
                if  p_visited[i][j]  and  a_visited[i][j]:
                    res.append([i,j])


        return res      
                
matrix=[ [1 ,  2 ,  2  , 3 , 5],[3 ,  2  , 3 , 4, 4],[2,   4 , 5,  3 ,  1],
        [6, 7,  1 ,  4,   5],[5,  1  , 1,   2 ,  4 ] ]      
if __name__ == "__main__":
    print(Solution().pacificAtlantic(matrix))                   
                    
        
#419. Battleships in a Board        
class Solution:
    def countBattleships(self, board):
        """
        :type board: List[List[str]]
        :rtype: int
        """
        m=len(board)
        if m==0:
            return 0
        n=len(board[0])
        
        count=0
        
        for i in range(m):
            for j in range(n):
                if board[i][j]=='.':
                    continue
                if i>0 and board[i-1][j]=='X':
                    continue
                if j>0 and board[i][j-1]=='X':
                    continue   
                count+=1
        return count
if __name__ == "__main__":
    print(Solution().countBattleships(board))      
        
#421. Maximum XOR of Two Numbers in an Array        
class Solution:
    def findMaximumXOR(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """        
# public int findMaximumXOR(int[] nums) {
#        int maxResult = 0; 
#        int mask = 0;
#        /*The maxResult is a record of the largest XOR we got so far. if it's 11100 at i = 2, it means 
#        before we reach the last two bits, 11100 is the biggest XOR we have, and we're going to explore
#        whether we can get another two '1's and put them into maxResult
#        
#        This is a greedy part, since we're looking for the largest XOR, we start 
#        from the very begining, aka, the 31st postition of bits. */
#        for (int i = 31; i >= 0; i--) {
#            
#            //The mask will grow like  100..000 , 110..000, 111..000,  then 1111...111
#            //for each iteration, we only care about the left parts
#            mask = mask | (1 << i);
#            
#            Set<Integer> set = new HashSet<>();
#            for (int num : nums) {
#                
#/*                we only care about the left parts, for example, if i = 2, then we have
#                {1100, 1000, 0100, 0000} from {1110, 1011, 0111, 0010}*/
#                int leftPartOfNum = num & mask;
#                set.add(leftPartOfNum);
#            }
#            
#            // if i = 1 and before this iteration, the maxResult we have now is 1100, 
#            // my wish is the maxResult will grow to 1110, so I will try to find a candidate
#            // which can give me the greedyTry;
#            int greedyTry = maxResult | (1 << i);
#            
#            for (int leftPartOfNum : set) {
#                //This is the most tricky part, coming from a fact that if a ^ b = c, then a ^ c = b;
#                // now we have the 'c', which is greedyTry, and we have the 'a', which is leftPartOfNum
#                // If we hope the formula a ^ b = c to be valid, then we need the b, 
#                // and to get b, we need a ^ c, if a ^ c exisited in our set, then we're good to go
#                int anotherNum = leftPartOfNum ^ greedyTry;
#                if (set.contains(anotherNum)) {
#                    maxResult= greedyTry;
#                    break;
#                }
#            }
#            
#            // If unfortunately, we didn't get the greedyTry, we still have our max, 
#            // So after this iteration, the max will stay at 1100.
#        }
#        
#        return maxResult;
#    }        
#        
        
        maxN=0
        mask=0
        for i in range(31,-1,-1):
            mask|=(1<<i)
            
            
             
            hashset=set()
            for num in nums:
               leftpart= (mask & num)
               hashset.add(leftpart)
            
            
            
            greedytry=maxN|(1<<i)
            
            for left in   hashset:
                anothernum=greedytry ^ left
                if anothernum in hashset:
                    maxN=greedytry
                    break
        return maxN
if __name__ == "__main__":
    print(Solution().findMaximumXOR([3, 10, 5, 25, 2, 8]))
                
#423. Reconstruct Original Digits from English
class Solution:
    def originalDigits(self, s):
        """
        :type s: str
        :rtype: str
        """
#        if (c == 'z') count[0]++;
#        if (c == 'w') count[2]++;
#        if (c == 'x') count[6]++;
#        if (c == 's') count[7]++; //7-6
#        if (c == 'g') count[8]++;
#        if (c == 'u') count[4]++; 
#        if (c == 'f') count[5]++; //5-4
#        if (c == 'h') count[3]++; //3-8
#        if (c == 'i') count[9]++; //9-8-5-6
#        if (c == 'o') count[1]++; //1-0-2-4
        count=[0 for _ in range(10)]
        for c in s:
            if c=='z':
                count[0]+=1
            if c=='w':
                count[2]+=1
            if c=='x':
                count[6]+=1
            if c=='s':
                count[7]+=1#7-6
            if c=='g':
                count[8]+=1
            if c=='u':
                count[4]+=1
            if c=='f':
                count[5]+=1#5-4
            if c=='h':
                count[3]+=1#3-8
            if c=='i':
                count[9]+=1#9-8-5-6
            if c=='o':
                count[1]+=1#1-0-2-4
                
        count[7] -= count[6]
        count[5] -= count[4]
        count[3] -= count[8]
        count[9] = count[9]-count[8]-count[5]-count[6]
        count[1] = count[1]-count[0]-count[2]-count[4]
        
        res=[]
        
        for i,n in enumerate(count):
            res+=[str(i)]*n
        return ''.join(res)
        
        
if __name__ == "__main__":
    print(Solution().originalDigits("fviefuro"))            
            
#424. Longest Repeating Character Replacement
class Solution:
    def characterReplacement(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        
        from collections import Counter
        counts=Counter()
        
        start=0
        end=0
        
        maxcount=0
        
        for end in range(len(s)):
            counts[s[end]]+=1
            
            maxcount=max(maxcount,counts[s[end]])
            
            if maxcount+k<end-start+1:
               counts[s[start]]-=1
               start+=1
        return len(s)-start
if __name__ == "__main__":
    print(Solution().characterReplacement("ABABAAA",2))                   
            
#433. Minimum Genetic Mutation            
class Solution(object):
    def minMutation(self, start, end, bank):
        """
        :type start: str
        :type end: str
        :type bank: List[str]
        :rtype: int
        """
        def validmutation(currentS,nextS):
            change=0  
            
            for i in range(len(currentS)):
                if currentS[i]!=nextS[i]:
                   change+=1
            return change==1
        
        from collections import deque
        
        dq=deque()
        dq.append([start,start,0])
        
        while dq:
            current,previous,Nstep=dq.popleft()
            
            if current==end:
                return Nstep
            
            for string in bank:
                if validmutation(current,string) and string!=previous:
                   dq.append([string,current,Nstep+1])
        return -1
bank= ["AAAACCCC", "AAACCCCC", "AACCCCCC"]

start="AAAAACCC"
end= "AACCCCCC"
if __name__ == "__main__":
    print(Solution().minMutation(start, end, bank))                 
                
#435. Non-overlapping Intervals            
# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: int
        """            
        if not intervals:
            return 0
        
        intervals.sort(key=lambda x:x.start)
        
        curentend=intervals[0].end
        count=0
        
        for x in intervals[1:]:
            if x.start<curentend:
                count+=1
                curentend=min(x.end,curentend)
            else:
                curentend=x.end
        return count
if __name__ == "__main__":
    print(Solution().minMutation(start, end, bank))              

#436. Find Right Interval           
# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    def findRightInterval(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[int]
        """

#        if not intervals:
#           return [] 
#
#        from collections import defaultdict
#
#        d=defaultdict(list)            
#            
#        for i,x in enumerate(intervals):
#            d[x.start]+=[i]
#        
#        
#        res=[]
#        for y in intervals:
#            if y.end in d:
#                print(y.end,d[y.end])
#                t=d[y.end][0]
#                res.append(t)
#                #d.pop(y.end)
#            elif not d:
#                res.append(-1)
#            else:
#                minv=float('inf')
#                
#                for z in d:
#                    if z-y.end >0 and z-y.end < minv:
#                       minv= z-y.end
#                       indx=d[z]
#                       v=z
#                if minv==float('inf'):
#                   res.append(-1)
#                else:
#                   print(indx)
#                   res.append(indx[0]) 
#                   #d.pop(v)
#        return res
#    
#intervals=[Interval(3,4),Interval(2,3),Interval(1,2)]
#
#intervals=[Interval(1,4),Interval(2,3),Interval(3,4)]  
#intervals=[Interval(1,2)]
#
#intervals=[Interval(4,5),Interval(2,3),Interval(1,2)] 
#
#intervals=[Interval(1,12),Interval(2,9),Interval(3,10),Interval(13,14),Interval(15,16),Interval(16,17)] 
#if __name__ == "__main__":
#    print(Solution().findRightInterval(intervals))                         
#            
#list_i=[[-100,-92],[-99,-49],[-98,-24],[-97,-38],[-96,-65],[-95,-22],[-94,-49],[-93,-14],
# [-92,-68],[-91,-81],[-90,-49],[-89,-23],[-88,5],[-87,-44],[-86,2],[-85,-81],[-84,-56],
# [-83,-53],[-82,-41],[-81,-68],[-80,-76],[-79,-9],[-78,-68],[-77,-19],[-76,-15],[-75,-41],
# [-74,26],[-73,6],[-72,-55],[-71,-35],[-70,28],[-69,-42],[-68,-55],[-67,1],[-66,-55],[-65,-31],
# [-64,16],[-63,-13],[-62,18],[-61,-39],[-60,8],[-59,14],[-58,36],[-57,-20],[-56,30],[-55,-9],
# [-54,-25],[-53,39],[-52,43],[-51,7],[-50,-48],[-49,5],[-48,-39],[-47,-2],[-46,23],[-45,46],
# [-44,-19],[-43,54],[-42,-11],[-41,-37],[-40,-17],[-39,28],[-38,12],[-37,-12],[-36,-34],[-35,19],
# [-34,44],[-33,-24],[-32,-3],[-31,3],[-30,69],[-29,53],[-28,8],[-27,-13],[-26,40],[-25,-10],[-24,40],
# [-23,14],[-22,4],[-21,49],[-20,-4],[-19,76],[-18,12],[-17,-15],[-16,2],[-15,81],[-14,-8],[-13,-8],
# [-12,40],[-11,88],[-10,79],[-9,15],[-8,-2],[-7,76],[-6,47],[-5,62],[-4,13],[-3,35],[-2,37],[-1,44],
# [0,2],[1,99],[2,74],[3,32],[4,42],[5,64],[6,84],[7,105],[8,103],[9,14],[10,20],[11,43],[12,58],
# [13,89],[14,50],[15,114],[16,59],[17,117],[18,87],[19,32],[20,81],[21,79],[22,117],[23,32],[24,120],
# [25,94],[26,40],[27,58],[28,35],[29,92],[30,73],[31,97],[32,115],[33,86],[34,102],[35,57],[36,132],
# [37,50],[38,110],[39,41],[40,131],[41,73],[42,81],[43,101],[44,61],[45,136],[46,87],[47,127],[48,84],
# [49,56],[50,123],[51,150],[52,148],[53,73],[54,109],[55,79],[56,146],[57,118],[58,64],[59,86],[60,84],
# [61,68],[62,76],[63,134],[64,103],[65,160],[66,87],[67,112],[68,135],[69,104],[70,97],[71,166],
# [72,136],[73,112],[74,119],[75,166],[76,127],[77,137],[78,102],[79,127],[80,166],[81,99],[82,155],
# [83,123],[84,132],[85,171],[86,183],[87,173],[88,112],[89,110],[90,135],[91,160],[92,128],[93,109],
# [94,120],[95,130],[96,139],[97,109],[98,178],[99,152]] 
#
#      
#intervals=[]
#for i in   list_i:
#     intervals.append(Interval(i[0],i[1]))     
            
        from bisect import bisect_left
        l=sorted([(e.start,i) for i,e in enumerate(intervals)])
        
        res=[]
        
        
        for e in intervals:
            r=    bisect_left(l,(e.end,))
            if r<len(intervals):
                res.append(l[r][1])
            else:
                res.append(-1)
        return res
            
        
#442. Find All Duplicates in an Array        
class Solution:
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        
        res=[]
        for num in nums:
            if nums[abs(num)-1]>0:
               nums[abs(num)-1]*=(-1)
            else:
               res.append(abs(num)) 
        return res
if __name__ == "__main__":
    print(Solution().findDuplicates([4,3,2,7,8,2,3,1]))       
        
        
#445. Add Two Numbers II        
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """        
        s1=[]
        s2=[]
        
        while l1:
            s1.append(l1.val)
            l1=l1.next
        while l2:
            s2.append(l2.val)
            l2=l2.next
            
            
            
        lnode=ListNode(0)
        
        sm=0
        while s1 or s2:
            if s1:
                sm=sm+s1.pop()
            if s2:
                sm=sm+s2.pop()
            
            lnode.val=sm%10
            
            sm=sm//10
            head=ListNode(sm)
            head.next=lnode
            lnode=head
        
        if sm==0:
            return head.next
        else:
            return head
        
#449. Serialize and Deserialize BST       
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        ls=[]
        def preorder(node):
            
            if node:
               ls.append(str(node.val))
               preorder(node.left)
               preorder(node.right)
        preorder(root)
        return ' '.join(ls)
            

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        def build(preorder,inorder):
            if inorder:
                idx=inorder.index(preorder.pop(0))
                n=TreeNode(inorder[idx])
                n.left=build(preorder,inorder[:idx])
                n.right=build(preorder,inorder[idx+1:])
                return n
        preorder = map(int, data.split())
        inorder = sorted(preorder)
        return build(preorder, inorder)

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))        
        
#450. Delete Node in a BST
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def deleteNode(self, root, key):
        """
        :type root: TreeNode
        :type key: int
        :rtype: TreeNode
        """
        if not root:
            return None
        
        if root.val < key:
            root.right=self.deleteNode(root.right,key)
        elif root.val >key:
            root.left=self.deleteNode(root.left,key)
        else:
            
            if not root.left:
                return root.right
            elif not root.right:
                return root.left
            else:
                
                smallestright=root.right
                while smallestright.left:
                      smallestright=smallestright.left
                
                
                smallestright.left=root.left
                return root.right
        return root
        
#451. Sort Characters By Frequency        
class Solution:
    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        from collections import Counter
        
        t=[c*n for c,n in Counter(s).most_common()]
        return ''.join(t)
#452. Minimum Number of Arrows to Burst Balloons
class Solution:
    def findMinArrowShots(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        if len(points)==0:
            return 0
        
        points.sort(key=lambda x:x[0])
    
        current_end=   points[0][1]  
        
        count=1
        for i in range(1,len(points)):
             if current_end>points[i][1]:
               current_end=   points[i][1] 
             if points[i][0]>current_end:
               count+=1
               current_end=points[i][1]
        return count
points=[[10,16], [2,8], [1,6], [7,12]] 
points=[[3,9],[7,12],[3,8],[6,8],[9,10],[2,9],[0,9],[3,9],[0,6],[2,8]]                
if __name__ == "__main__":
    print(Solution().findMinArrowShots(points))       
                        
#454. 4Sum II                
class Solution:
    def fourSumCount(self, A, B, C, D):
        """
        :type A: List[int]
        :type B: List[int]
        :type C: List[int]
        :type D: List[int]
        :rtype: int
        """
        from collections import defaultdict
        dd=defaultdict(int)
        
        for a in A:
            for b in B:
                dd[a+b]+=1
        
        res=0
        for c in C:
            for d in D:
                remainder=0-c-d
                if remainder in dd:
                    res+=dd[remainder]
            
        return res
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]
if __name__ == "__main__":
    print(Solution().fourSumCount(A, B, C, D))       
                
#456. 132 Pattern                
class Solution:
    def find132pattern(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if not nums:
           return False     
        e3=float('-inf')
        st=[]
        
        for e in reversed(nums):
            if e<e3:
                return True
            while st and e>st[-1]:
                e3=st.pop()
            st.append(e)
        return False
if __name__ == "__main__":
    print(Solution().find132pattern([3, 1, 4, 2]))            
                
#457. Circular Array Loop                
class Solution(object):
    def circularArrayLoop(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        visited=set()
        from collections import defaultdict
        m= defaultdict(int)     
        size=len(nums)   
        
        for i in range( size):
            
            if i in visited:
               continue
            
            cur=i
            while True:
                visited.add(cur)
                nextone=(nums[cur]+cur+size)%size
                if nextone==cur or nums[nextone]*nums[cur]<0:
                   break
                
                if m[nextone]>0:
                    return True
                m[cur]+=1
                cur=nextone
        return False
if __name__ == "__main__":
    print(Solution().circularArrayLoop( [-1, 2]))                
                
#462. Minimum Moves to Equal Array Elements II                
class Solution:
    def minMoves2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()

        i=0
        j=len(nums)-1
        count=0
        while i<j:
              count+=nums[j]-nums[i]    
              j-=1
              i+=1
        return count
if __name__ == "__main__":
    print(Solution().minMoves2( [1,2,3]))                
#464. Can I Win               
class Solution:
    def canIWin(self, maxChoosableInteger, desiredTotal):
        """
        :type maxChoosableInteger: int
        :type desiredTotal: int
        :rtype: bool
        """
        if  sum(range(1,maxChoosableInteger +1))<      desiredTotal:
            return False
        
        memo={}
        
        def dfs(nums,desiredTotal):
            
            hashn=str(nums)
            
            if hashn in memo:
                return memo[hashn]
            
            if nums[-1] >=desiredTotal:
                return True
            
            for i in range(len(nums)):
                if not dfs(nums[:i]+nums[i+1:],desiredTotal-nums[i]):
                    memo[hashn]=True
                    return True
            memo[hashn]=False
        
            return False
        
        return dfs(list(range(1,maxChoosableInteger+1)),desiredTotal)
        
if __name__ == "__main__":
    print(Solution().canIWin( 10,11))      
        
#467. Unique Substrings in Wraparound String        
class Solution:
    def findSubstringInWraproundString(self, p):
        """
        :type p: str
        :rtype: int
        """

        if not p:
            return 0
        
        maxlenth=0
        
        count=[0 for _ in range(26)]
        
        for i in range(len(p)):
            if i>0 and (ord(p[i])-ord(p[i-1])==1 or ord(p[i])-ord(p[i-1])==-25):
               maxlenth+=1
            else:
               maxlenth=1
               
            idx=ord(p[i])-ord('a')
            count[idx]=max(maxlenth,count[idx])
        print(count)
        return sum(count)
if __name__ == "__main__":
    print(Solution().findSubstringInWraproundString( "zaba"))      
                
#468. Validate IP Address        
class Solution:
    def validIPAddress(self, IP):
        """
        :type IP: str
        :rtype: str
        """
        if not IP:
            return "Neither"
        
        def isIPv4(IP):
            l= IP.split('.')
            
            if len(l)!=4:
                return "Neither"
            for digit in l:
                if not (digit.isdigit() and int( digit ) < 256 and str(int( digit ))==digit):
                   return "Neither"
            return "IPv4"
        
        def isIPv6(IP):
            import string
            l= IP.split(':')
            if len(l)!=8:
                return "Neither"
            
            for digit in l:
                if not (  digit.isalnum()  and len(digit) <5 and len(digit) >0):
                    return "Neither"
                if not set(digit).issubset(set('0123456789abcdefABCDEF')):
                   return "Neither"
                    
            return "IPv6"
        
        if '.' in IP:
            return isIPv4(IP)
        elif ':' in IP:
            return isIPv6(IP)
        else:
            return "Neither"
IP='172.16.254.1'
IP='2001:0db8:85a3:0000:0000:8a2e:0370:7334'
IP='2001:0db8:85a3::8A2E:0370:7334'
IP='02001:0db8:85a3:0000:0000:8a2e:0370:7334'
IP="20EE:FGb8:85a3:0:0:8A2E:0370:7334"
if __name__ == "__main__":
    print(Solution().validIPAddress(IP))           
        
#473. Matchsticks to Square        
class Solution:
    def makesquare(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
      
        if sum(nums)%4!=0:
           return False
       
        if len(nums)<4:
            return False
       
        target=sum(nums)//4
       
        targets=[target for _ in range(4)] 
        nums.sort(reverse=True)
        def dfs(nums,targets,pos):
           if pos==len(nums):
              return True 
           visited=set()
           for i in  range(4):
               if targets[i]>=nums[pos] and targets[i] not in visited:
                  targets[i]-=nums[pos]
                  if dfs(nums,targets,pos+1):
                      return True
                  visited.add(targets[i])
                  targets[i]+=nums[pos]
           return False
        return dfs(nums,targets,0)
if __name__ == "__main__":
    print(Solution().makesquare([12,12,12,12,12,12,12,12,12,12,12,12,12]))       
       
#474. Ones and Zeroes       
class Solution:
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        #accepted solution for Python 2   not for Python3
        if not strs:
            return 0
        
        def counting(s):
            from collections import Counter
            return (s.count('0'),s.count('1'))
        dp=[[0 for _ in range(n+1)] for _ in range(m+1)]
        
        for z,o in [counting(s)  for s in strs]:
            #print(z,o)
            for x in range(m,-1,-1):
                for y in range(n,-1,-1):
                   if x>=z and y>=o:
                     dp[x][y]=max(dp[x][y],dp[x-z][y-o]+1)
        return dp[m][n]
        
        
        
strs=["10", "0001", "111001", "1", "0"]
strs=["10","0","1"]
m=5
n=3
if __name__ == "__main__":
    print(Solution().findMaxForm(strs, m, n))           
        
#477. Total Hamming Distance       
class Solution:
    def totalHammingDistance(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        s= [b.count('0')*b.count('1') for  b in  zip(*map('{:032b}'.format,nums) ) ]
        return sum(s)
nums=[4, 14, 2]
if __name__ == "__main__":
    print(Solution().totalHammingDistance(nums))       
    
       
#481. Magical String       
class Solution:
    def magicalString(self, n):
        """
        :type n: int
        :rtype: int
        """
   
        if n<=0:
            return 0
        if n<=3:
            return 1
        
        A=[0 for _ in range(n+1)]
        A[0]=1
        A[1]=A[2]=2
        
        head=2
        tail=3
        num=1
        result=1
        
        while tail < n:
            for i in range(A[head]):
                A[tail]=num
                if num==1 and tail<n:
                    result+=1
                tail+=1
            head+=1
            num=num^3
        #print(A)
        return result
if __name__ == "__main__":
    print(Solution().magicalString(6))              
        
#486. Predict the Winner        
class Solution:
    def PredictTheWinner(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        self.memo={}   
        
        def dfs(s,e,nums):
            if (s,e) not in self.memo:
              if s==e:
                 self.memo[(s,e)]=nums[s]
              else:
                 self.memo[(s,e)]=max(-dfs(s+1,e,nums)+nums[s],nums[e]-dfs(s,e-1,nums))
            return self.memo[(s,e)] 
        return dfs(0,len(nums)-1,nums)>=0
if __name__ == "__main__":
    print(Solution().PredictTheWinner([1,5,2]))         
        
#491. Increasing Subsequences        
class Solution:
    def findSubsequences(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if not nums or len(nums)==1:
            return []
        
        subs={()}
        
        for num in nums:
#            for sub in subs:
#                if not sub or sub[-1]<=nums[i]:
#                    sub+=(nums[i],)
#                    subs|={sub}
#            RuntimeError: Set changed size during iteration
            
            subs|={sub+(num,) for sub in subs if not sub or sub[-1]<=num}
        return [sub for sub in subs if len(sub)>1]
       
  
    
        
    
    
nums=   [4, 6, 7, 7]
if __name__ == "__main__":
    print(Solution().findSubsequences(nums))         
                    
#494. Target Sum            
class Solution:
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
#        n=len(nums)
#        self.res=0
#        def dfs(nums,path):
#            if sum(nums) +sum(path) < S or sum(path)-sum(nums)>S:
#                return 
#            if len(path)==n and sum(path)==S:
#                self.res+=1
#                return 
#            
#            for i in range(len(nums)):
#                
#                dfs(nums[i+1:],path+[nums[i]])
#                dfs(nums[i+1:],path+[nums[i]*(-1)])
#        dfs(nums,[])
#        return self.res
#    
    
    
        dp={0:1}
        from collections import defaultdict
        for num in nums:
            nextdp=defaultdict(int)
            for s,n in dp.items():
                if n>0:
                    nextdp[s+num]+=n
                    nextdp[s-num]+=n
            dp=nextdp
        return dp[S]
    
    
    
    
    
if __name__ == "__main__":
    print(Solution().findTargetSumWays([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],0))               
        
#495. Teemo Attacking        
class Solution:
    def findPoisonedDuration(self, timeSeries, duration):
        """
        :type timeSeries: List[int]
        :type duration: int
        :rtype: int
        """
    
        
        if not timeSeries or not duration:
            return 0
        
        n=len(timeSeries)
        
        res=0
        for i in range(1,n):
            if timeSeries[i]-timeSeries[i-1]>=duration:
                res+=duration
                
            else:
                res+=(timeSeries[i]-timeSeries[i-1])
        return res+duration
timeSeries=[1,4]
duration=2
if __name__ == "__main__":
    print(Solution().findPoisonedDuration(timeSeries, duration))           
#498. Diagonal Traverse       
class Solution:
    def findDiagonalOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix:
            return []
        m=len(matrix)
        n=len(matrix[0])
        from collections import defaultdict
        dd=defaultdict(list)
        
        
        for i in range(m):
            for j in range(n):
                dd[i+j+1].append(matrix[i][j])
        res=[]
        for k in dd.keys():
            print(k,dd[k])
            if k%2==1:
                res+=reversed(dd[k])
            else:
                res+=dd[k]
        return res
                
            
        
matrix=[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
if __name__ == "__main__":
    print(Solution().findDiagonalOrder( matrix)  )             
                
#503. Next Greater Element II       
class Solution:
    def nextGreaterElements(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return []
        res=[-1 for _ in range(len(nums))]
        stack=[]
        
        for i in range(len(nums)):
            while stack and nums[stack[-1]]<nums[i]:
                res[stack.pop()]=nums[i]
            stack.append(i)
        if stack:
            
            for i in range(stack[0]+1):
                while stack and nums[stack[-1]]<nums[i]:
                      res[stack.pop()]=nums[i]
                stack.append(i)
        return res
if __name__ == "__main__":
    print(Solution().nextGreaterElements( [1,2,1])  ) 

#508. Most Frequent Subtree Sum
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def findFrequentTreeSum(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        
        def dfsSum(node,d):
            
            if not node:
                return 
            if not node.left and not node.right:
                sumnode=node.val
            elif not node.left:
                sumnode=node.val+dfsSum(node.right,d)
            elif not node.right:
               sumnode=node.val+dfsSum(node.left,d)
            elif  node.right and node.right:
               sumnode=node.val+dfsSum(node.left,d)+dfsSum(node.right,d)
            d[sumnode]+=1
            return sumnode
        
        from collections import defaultdict
        d=defaultdict(int)
        ans=[]
        mostfreq=0
        dfsSum(root,d)
        for k,f in d.items():
            if f> mostfreq:
                mostfreq=f
                ans=[k]
            elif f==mostfreq:
                ans.append(k)
        return ans

t=TreeNode(5)       
t.left= TreeNode(2) 
t.right= TreeNode(-5)                
if __name__ == "__main__":
    print(Solution().findFrequentTreeSum( t  ) )        
        
#513. Find Bottom Left Tree Value        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        treeq=[ root]
        
        
        while treeq:
            tempq=[]
            
            ans=treeq[0].val
            
            for treenode in treeq:
                if treenode.left:
                   tempq.append(treenode.left)
                if treenode.right:
                   tempq.append(treenode.right)  
            treeq=tempq
        return ans
t=TreeNode(2)       
t.left= TreeNode(1) 
t.right= TreeNode(3)  
t.right.left=  TreeNode(5)                 
if __name__ == "__main__":
    print(Solution().findBottomLeftValue( t  ) )        
        
#515. Find Largest Value in Each Tree Row        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def largestValues(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """

        if not root:
            return []
            
        maxofrow=[]   
        row=[root]
        
        while row:
            
            maxofrow.append(max(x.val for x in row))
            
            row2=[]
            for node in row:
               if node.left:
                  row2.append(node.left)
        
               if node.right:
                  row2.append(node.right)
                
            row=row2
        return maxofrow
if __name__ == "__main__":
    print(Solution().largestValues( t  ) )        
#516. Longest Palindromic Subsequence        
class Solution:
    def longestPalindromeSubseq(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        
        
        
        n=len(s)
        if s==s[::-1]:
            return n
        
        dp=[[0 for _ in range(n)] for _ in range(n)]
        
        
        for i in range(n-1,-1,-1):
            dp[i][i]=1
            for j in range(i+1,n):
                if s[i]==s[j]:
                    dp[i][j]=dp[i+1][j-1]+2
                else:
                    dp[i][j]=max(dp[i+1][j],dp[i][j-1])
        return dp[0][n-1]
if __name__ == "__main__":
    print(Solution().longestPalindromeSubseq( "bbbab" ) )          
        
#518. Coin Change 2        
class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        
        if not amount :
            return 1
        if not coins:
            return 0
        n=len(coins)
        
#        self.res=[]
#        def backtrack(coins,path,amount):
#            if amount==0 and sorted(path) not in self.res:
#               self.res.append(sorted(path))
#               return 
#            if amount>0:
#                for coin in coins:
#                    backtrack(coins,path+[coin],amount-coin)
#            if amount<0:
#                return 
#        backtrack(coins,[],amount)
#        #print(self.res)
#        return len(self.res)
        
        dp=[0 for _ in range(amount+1)]
        dp[0]=1
        
        for coin in coins:
            for i in range(coin,amount+1):
                dp[i]+=dp[i-coin]
        return dp[amount]
        
        
        
amount=500
coins=[1,2,5]
if __name__ == "__main__":
    print(Solution(). change( amount, coins) )       
        
#522. Longest Uncommon Subsequence II        
class Solution:
    def findLUSlength(self, strs):
        """
        :type strs: List[str]
        :rtype: int
        """
        def isSubsequence(s1,s2):
            s2=iter(s2)
            
            return all( c in s2 for c in s1)
        if not strs:
            return 0
        strs.sort(key=len,reverse=True)
        
        for s in strs:
            if sum( isSubsequence(s,s2) for s2 in strs)==1:
                return len(s)
        return -1
       
s1= "aba"
s2= "cdc"      
if __name__ == "__main__":
    print(Solution().findLUSlength( ["aba", "cdc", "eae"] ))
             
#523. Continuous Subarray Sum        
class Solution:
    def checkSubarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        if not nums:
            return False
        
        n=len(nums)
        if k==0:
            for i in range(1,n):
                if nums[i]==0 and nums[i-1]==0:
                   return True
            return False
        
        sumN=list(nums)
        
        for i in range(1,n):
          sumN[i]+= sumN[i-1]
        
        for i in range(n-1):
            for j in range(i+1,n):
                
                if sumN[j]%k==0:
                    return True
                if j-i>=2 and (sumN[j]  -sumN[i])%k==0:
                    return True
        return False
nums=[23,2,4,6,7]
if __name__ == "__main__":
    print(Solution().checkSubarraySum(nums,-6 ))   


#524. Longest Word in Dictionary through Deleting        
class Solution:
    def findLongestWord(self, s, d):
        """
        :type s: str
        :type d: List[str]
        :rtype: str
        """
        if not s or not d :
            return ''
        
        res=[]
        
        def isSubsequence(s1,s2):
            s2=iter(s2)
            
            return all(c in s2 for c in s1)
        
        d.sort(key=len,reverse=True)
        
        for s1 in d:
            
            if isSubsequence(s1,s):
               if not res:
                res.append(s1)
               elif len(s1)<len(res[0]):
                   break
               else:
                   res.append(s1)
        
        if not res:
            return ''
        else:
            res.sort()
            return res[0] 
d =  ["a","b","c"]
s = "abpcplea"
if __name__ == "__main__":
    print(Solution().findLongestWord(s, d ))                
        
#525. Contiguous Array        
class Solution:
    def findMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        if not nums:
            return 0
        
        maxlen=0
        d={0:0}
        count=0
        for i,num in enumerate(nums,1):
            if num==0:
                count+=1
            elif num==1:
                count-=1
            
            if count not in d:
                d[count]=i
            else:
                print(d)
                print(i)
                maxlen=max(maxlen,i-d[count])
        return maxlen
if __name__ == "__main__":
    print(Solution().findMaxLength([0,1] ))            
            
#526. Beautiful Arrangement           
class Solution:
    def countArrangement(self, N):
        """
        :type N: int
        :rtype: int
        """
        self.res=0
        visited=[False for _ in range(N+1)]
        
        def backtrack(N,pos,visited):
            if pos <1:
                self.res+=1
                return 
            for i in range(1,N+1):
                if  not visited[i] and (pos%i==0 or i%pos==0):
                   visited[i]=True
                   backtrack(N,pos-1,visited)
                   visited[i]=False
        backtrack(N,N,visited)
        return self.res
if __name__ == "__main__":
    print(Solution().countArrangement(5))            
                            
                
#529. Minesweeper            
class Solution:
    def updateBoard(self, board, click):
        """
        :type board: List[List[str]]
        :type click: List[int]
        :rtype: List[List[str]]
        """
        if  not board:
            return []
        if not click:
            return board
        
        click=tuple(click)
        
        R=len(board)
        C=len(board[0])
        
        def neighbor(r,c):
            for dr in range(-1,2,1):
                for dc in range(-1,2,1):
                    if (dr or dc) and  0<=r+dr and  0<=c+dc and R>r+dr and C>c+dc:
                        yield r+dr,c+dc
        
        
        
        seen={click}
        stack=[click]
        
        while stack:
             r,c= stack.pop()
             seen.add((r,c))
             
             if board[r][c]=='M':
                board[r][c]='X'
               
             else:
                 sumnei=sum(  1 for r1,c1 in neighbor(r,c) if  board[r1][c1]  in 'MX')
                 if sumnei:
                     board[r][c]=str(sumnei)
                 else:
                      board[r][c]='B'
                      for  r1,c1 in neighbor(r,c):
                           if board[r1][c1]  in 'E'  and (r1,c1) not in seen:
                               stack.append((r1,c1))
        return board
board= [['E', 'E', 'E', 'E', 'E'],
 ['E', 'E', 'M', 'E', 'E'],
 ['E', 'E', 'E', 'E', 'E'],
 ['E', 'E', 'E', 'E', 'E']]
click=[1,2]                              
if __name__ == "__main__":
    print(Solution().updateBoard( board, click))                          
        
board[1][2]        
#535. Encode and Decode TinyURL        
class Codec:
    import string
    aphnum=string.ascii_letters+'0123456789'
    
    
    def __init__(self):
        self.codes={}
        self.url={}

    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.
        
        :type longUrl: str
        :rtype: str
        """
        import random
        while longUrl not in self.url:
              code = ''.join([random.choice(Codec.aphnum) for _ in range(6)])
              if code not in self.codes:
                 self.codes[code]=longUrl
                 self.url[longUrl]=code
        
        return 'http://tinyurl.com/'+ self.url[longUrl]

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.
        
        :type shortUrl: str
        :rtype: str
        """
        return  self.codes[shortUrl[-6:]]
        
codec = Codec()
url='https://leetcode.com/problems/design-tinyurl'
# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(url))        
        
        
#537. Complex Number Multiplication        
class Solution:
    def complexNumberMultiply(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        a1 ,a2= a.split('+')   
        b1 ,b2= b.split('+')
        
        c1=int(a1) * int(b1)+(int(a2[:-1]) * int( b2[:-1]))*(-1)
        c2= int(a1) *int( b2[:-1])+int(b1) *int( a2[:-1])
        
      
        
        return str(c1) + '+'+str(c2)+'i'
a="1+1i"
b="1+1i"
if __name__ == "__main__":
    print(Solution().complexNumberMultiply( a,b))        
        
#539. Minimum Time Difference        
class Solution:
    def findMinDifference(self, timePoints):
        """
        :type timePoints: List[str]
        :rtype: int
        """
     
        
        a1,a2=map(int,timePoints[0].split(':'))
        b1,b2=map(int,timePoints[1].split(':'))
        
        t=[]
        for time in timePoints:
            temp=list(map(int,time.split(':')))
            t.append(temp[0]*60+ temp[1])
        
        t.sort()
        t2=t[1:]+[t[0]+24*60]
      
       
        return min(y-x for x,y in zip(t,t2))
timepoints=["00:00","23:59","00:00"]
if __name__ == "__main__":
    print(Solution().findMinDifference( timepoints))                
        
#540. Single Element in a Sorted Array        
class Solution:
    def singleNonDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """        
        lo=0
        hi=len(nums)-1
        
        while lo<hi:
            mid=(lo+hi)//2
            if nums[mid]==nums[mid^1]:
                lo=mid+1
            else:
                hi=mid
        return nums[lo]
        
#542. 01 Matrix        
class Solution:
    def updateMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """        
        R=len(matrix)
        C=len(matrix[0])
        
        def neighbor(r,c):
            for dr, dc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if  dr >= 0  and dr < R and dc >= 0  and dc < C:
                        yield dr,dc
        
        from collections import deque
        q=deque([((x,y),0)  for x in range(R) for y in range(C) if matrix[x][y]==0  ])
        #print(q)
        seen={x for x,_ in q}
        print(seen)
        ans=[[0 for _ in range(C)] for _ in range(R)]
        
        while q:
            (x,y),depth=q.popleft()
            print(depth)
            ans[x][y]=depth
            for i,j in neighbor(x,y):
                if (i,j) not in seen:
                    seen.add((i,j))
                    #print(i,j)
                    q.append(((i,j),depth+1))
        return ans
    
matrix=[[    0 ,0, 0],
[0, 1, 0],
[1, 1 ,1]]
if __name__ == "__main__":
    print(Solution().updateMatrix( matrix))                      
        
#547. Friend Circles        
class Solution:
    def findCircleNum(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        def dfs(M,visited,person):
            for other in range(len(M)):
                if M[person][other]==1 and not visited[other]:
                     visited[other]=True
                     dfs(M,visited,other)
        count=0
        visited=[False for _ in range(len(M))]
        for i in range(len(M)):
            if not visited[i]:
                dfs(M,visited,i)
                count+=1
        return count
            
M=[[1 ,1, 0],
[1, 1, 1],
[0, 1 ,1]]        
if __name__ == "__main__":
    print(Solution().findCircleNum( M))          
        
#553. Optimal Division        
class Solution:
    def optimalDivision(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        A=list(map(str,nums))
        if len(A)<=2:
            return '/'.join(A)
        return '{}/({})'.format(A[0],'/'.join(A[1:]))
if __name__ == "__main__":
    print(Solution().optimalDivision( [1000,100,3]))         
#554. Brick Wall
class Solution:
    def leastBricks(self, wall):
        """
        :type wall: List[List[int]]
        :rtype: int
        """
        
        if not wall:
            return 0
        
        m=len(wall)
        wallsum=[list(wall[i]) for i in range(m)]
        for i in range(m):
            for j in range(1,len(wall[i])):
              
             
               wallsum[i][j]=wallsum[i][j-1]+wallsum[i][j]
        res=[]
        for x in range(m):
            res+=wallsum[x][:-1]
        from collections import Counter
        count=Counter(res)
        
        if not count:
            return m
        return abs(max (count.values()) -m      )
wall=[[1,2,2,1],
 [3,1,2],
 [1,3,2],
 [2,4],
 [3,1,2],
 [1,3,1,1]] 

wall=[[1],[1],[1]]  

if __name__ == "__main__":
    print(Solution().leastBricks( wall))
        
        
#556. Next Greater Element III        
class Solution:
    def nextGreaterElement(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n<11:
            return -1
        
        nlist=[]
        while n:
            nlist.append(n%10)
            n=n//10
            
        nlist=nlist[::-1]
        
        a=len(nlist)
        
        k=a-2
        while k>=0 and nlist[k]>=nlist[k+1]:
            k-=1
        if k<0:
            return -1
        
        for i in range(a-1,0,-1):
            #print(nlist[i],nlist[i-1])
            if nlist[i] > nlist[i-1]:
                break
        
       
        smallest=i-1
        
        smaller=smallest+1
        for j in range( smallest+2,a):
            if nlist[j]<=  nlist[smaller]  and nlist[j]>nlist[smallest]:
               smaller=j
        
        nlist[smallest], nlist[smaller]= nlist[smaller],nlist[smallest]
        
        res=nlist[:smallest+1]+nlist[smallest+1:][::-1]
        
        res2=list(map(str,res))
            
        return int(''.join(res2)) if int(''.join(res2))<  2147483647 else -1

n=1999999999     
if __name__ == "__main__":
    print(Solution().nextGreaterElement(n))  
        
#560. Subarray Sum Equals K       
class Solution:
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """        
        if not nums:
            return 0
        n=len(nums)
        
        
        from collections  import defaultdict
        
        hashmap=defaultdict(int)
        hashmap[0]=1
        
        res=0
        sumn=0
        for i in range(n):
             sumn+=nums[i]
             
             if (sumn-k) in  hashmap :
                 res+= hashmap[sumn-k]
             hashmap[sumn]+=1
            
        return res
if __name__ == "__main__":
    print(Solution().subarraySum([0,0,0,0,0,0,0,0,0,0],0))            
            
#565. Array Nesting        
class Solution:
    def arrayNesting(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        self.visited=[False for _ in range(len(nums))]
        
        self.maxcount=0
        
        def dfs(nums,i,count):
            
            if not self.visited[i]:
                print(i)
                j=nums[i]
                count+=1
                if self.maxcount<count:
                   self.maxcount=count
                self.visited[i]=True
                dfs(nums,j,count)
                
        for i in range(len(nums)):
            if not self.visited[i]: 
               
               dfs(nums,i,0)
               
        
        return self.maxcount
        
if __name__ == "__main__":
    print(Solution().arrayNesting(nums)) 

           
class Solution:
    def arrayNesting(self, nums):            
           
        ans=0
        step=0
        seen=[False for _ in range(len(nums))]
        
        
        for i in range(len(nums)):
            
            while not seen[i]:
                seen[i]=True
                i=nums[i]
                step+=1
                ans=max(ans,+step)
            step=0
        return ans
                
if __name__ == "__main__":
    print(Solution().arrayNesting([5,4,0,3,1,6,2])  )                      
            
        
#567. Permutation in String        
class Solution:
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        n1=len(s1)    
        n2=len(s2)
        
        ss1=[ord(x)-ord('a') for x in s1]
        ss2=[ord(x)-ord('a') for x in s2]
        
        target=[0 for _ in range(26)]
        for x in ss1:
            target[x]+=1
        
        window=[0 for _ in range(26)]
        
        for i,y in enumerate(ss2):
            window[y] +=1
            
            if i>=n1:
              window[ss2[i-n1]]-=1
            
            if window==target:
                return True
        
        return False
    
s1 = "adc" 
s2 = "dcda"
if __name__ == "__main__":
    print(Solution().checkInclusion(s1, s2)  )                      
                    
#576. Out of Boundary Paths        
class Solution:
    def findPaths(self, m, n, N, i, j):
        """
        :type m: int
        :type n: int
        :type N: int
        :type i: int
        :type j: int
        :rtype: int
        """
        if N <0:
            return 0
        
        count=[[0 for _ in range(n)]  for _ in range(m)]
        count[i][j]=1
        result=0
        mod=1000000007
        
        dir={(1,0),(-1,0),(0,1),(0,-1)}
        
        for step in range(N):
            
            temp=[[0 for _ in range(n)]  for _ in range(m)]
            
            for r in range(m):
                for c in range(n):
                    for step in dir:
                        nc=c+step[1]
                        nr=r+step[0]
                        if nc <0 or nc>=n or nr<0 or nr>=m:
                            result=(result+count[r][c])%mod
                        else:
                            temp[nr][nc]=( temp[nr][nc]+count[r][c])%mod
            count=temp
        return result
    
    
m = 1
n = 3
N = 3
i = 0
j = 1     
if __name__ == "__main__":
    print(Solution().findPaths( m, n, N, i, j)  ) 
        
#583. Delete Operation for Two Strings  
class Solution:
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        def findLongeststring(word1, word2):
            m=len(word1)
            n=len(word2)
            dp=[[0 for _ in range(n+1)]  for _ in range(m+1)]
            
            res=0
            
            for i in range(m):
                for j in range(n):
                    if word1[i]==word2[j]:
                        dp[i+1][j+1]= dp[i][j]+1
                    else:
                        dp[i+1][j+1]=max( dp[i][j+1],dp[i+1][j])
                    res=max(dp[i+1][j+1], res)
            return dp[m][n]
        
        if not word1 or not word2:
            return len(word1) or len(word2)
        
        longestString=findLongeststring(word1, word2)
        
        return len(word1+word2)-2*longestString
word1="sea"
word2=""        
if __name__ == "__main__":
    print(Solution().minDistance( word1, word2)  )        
        
#592. Fraction Addition and Subtraction        
class Solution:
    def fractionAddition(self, expression):
        """
        :type expression: str
        :rtype: str
        """
        
        import re
        import math
        integers=list( map(int,re.findall('[+-]?\\d+',expression)) )   
       
        A,B=0,1
       
        for a,b in zip(integers[::2],integers[1::2]):
           A=A*b+B*a
           B=B*b
           
           g=math.gcd(A,B)
           A=A//g
           B=B//g
         
        return '{0}/{1}'.format(A,B)
expression="1/3-1/2"
if __name__ == "__main__":
    print(Solution().fractionAddition( expression)  )             
           
#593. Valid Square        
class Solution:
    def validSquare(self, p1, p2, p3, p4):
        """
        :type p1: List[int]
        :type p2: List[int]
        :type p3: List[int]
        :type p4: List[int]
        :rtype: bool
        """
        

        def distance(A,B):      
            return (A[0]-B[0])**2+(A[1]-B[1])**2
        
        dis=set()
        
        dis.add(distance(p1,p2))
        dis.add(distance(p1,p3))
        dis.add(distance(p1,p4))
        dis.add(distance(p2,p3))
        dis.add(distance(p2,p4))
        dis.add(distance(p3,p4))
        
        
        return 0 not in dis and len(dis)==2

p1 = [0,0]
p2 = [1,1]
p3 = [1,0]
p4 = [0,1]
        
if __name__ == "__main__":
    print(Solution().validSquare( p1, p2, p3, p4)  )        
        
#609. Find Duplicate File in System        
class Solution:
    def findDuplicate(self, paths):
        """
        :type paths: List[str]
        :rtype: List[List[str]]
        """
  

        from collections import defaultdict
        
        M=defaultdict(list)
        directory=[]
        file=[]
        for path in paths:
             data=path.split()
             root=data[0]
             files=data[1:]
             
             for file in files:
                 name,_,content=file.partition('(')
                 M[content[:-1]].append(root+'/'+name)
        return [x for x in M.values() if len(x)>1]
paths=["root/a 1.txt(abcd) 2.txt(efgh)", "root/c 3.txt(abcd)", "root/c/d 4.txt(efgh)", "root 4.txt(efgh)"]                 
if __name__ == "__main__":
    print(Solution(). findDuplicate(paths)  )        
                
#611. Valid Triangle Number        
class Solution:
    def triangleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        if len(nums)<3:
            return 0
        
#        def validation(a,b,c):
#            
#            if a and b and c and a+b>c :
#                return True
#            else:
#                return False
        nums.sort()
        n=len(nums)
        count=0
#        for i in range(n):
#            for j in range(i+1,n):
#                for k in range(j+1,n):
#                    if validation(nums[i],nums[j],nums[k]):
#                        count+=1
#                    else:
#                        continue
        
        for i in range(n-1,1,-1):
            l=0
            r=i-1
            while l<r:
                
                if nums[l]+nums[r]>nums[i]:
                    count+=r-l
                    r-=1
                else:
                    l+=1
        return count
                    
      
nums=[2,3,4,9,10,11]    
if __name__ == "__main__":
    print(Solution().triangleNumber( nums)  )        

#621. Task Scheduler
class Solution:
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        mapcnt=[0 for _ in range(26)]
        time=0
        i=0
        
        for x in tasks:
            mapcnt[ord(x)-ord('A')]+=1
        mapcnt.sort()
        
        while mapcnt[25]>0:
        
            i=0
            while i<=n:
                
                if mapcnt[25]==0:
                    break
                if i<26 and  mapcnt[25-i]>0:
                   mapcnt[25-i]-=1
                time+=1
                i+=1
            mapcnt.sort()
        return time

tasks=["A","A","A","B","B","B"]
n=2
if __name__ == "__main__":
    print(Solution().leastInterval( tasks, n)  ) 
        
#623. Add One Row to Tree 
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
            
class Solution:
    def addOneRow(self, root, v, d):
        """
        :type root: TreeNode
        :type v: int
        :type d: int
        :rtype: TreeNode
        """
        if d==1:
           n=TreeNode(v)
           n.left=root
           return n
       
        from collections import deque
        q=deque()
        q.append(root)
        depth=1
        while depth < d-1:
            temp=deque()
            while q:
                node=q.popleft()
                if node.left:
                    temp.append(node.left)
                if node.right:
                    temp.append(node.right)
            depth+=1
            q=temp
        print(q)
        while q:
            node=q.popleft()
            temp=node.left
            node.left=TreeNode(v)
            node.left.left=temp
            
            temp=node.right
            node.right=TreeNode(v)
            node.right.right=temp
        return root
    
    
root=TreeNode(4)       
root.left= TreeNode(2) 
root.left.right= TreeNode(1) 
root.left.left= TreeNode(3)                
d=3
v=1          
if __name__ == "__main__":
    printT(Solution().addOneRow( t,v, d)  ) 


from collections import defaultdict,deque
def printT(root):
    p=deque()
    p.append(root)
    while p:
       node=p.popleft()
       print(node.val)
       if node.left:
         p.append(node.left)
       if node.right:
         p.append(node.right)      
           
#626. Exchange Seats           
#+---------+---------+
#|    id   | student |
#+---------+---------+
#|    1    | Abbot   |
#|    2    | Doris   |
#|    3    | Emerson |
#|    4    | Green   |
#|    5    | Jeames  |
#+---------+---------+
#select s1.id-1 as id , s1.student as student
#from seat s1
#where s1.id mod 2 =0 
#union
#select s2.id+1 as id , s2.student as student
#from seat s2
#where s2.id mod 2 =1 and s2.id!= ( select max(id) from seat  ) 
#union                    
#select s3.id as id , s3.student as student
#from seat s3
#where s3.id mod 2 =1 and s3.id= ( select max(id) from seat  ) 
#                     
#order by id asc;

#636. Exclusive Time of Functions           
class Solution:
    def exclusiveTime(self, n, logs):
        """
        :type n: int
        :type logs: List[str]
        :rtype: List[int]
        """           

        stack=[]
        ans=[0 for _ in range(n)]
        prev_time=0
        
        for log in logs:
            fn,typ,time=log.split(':')
            fn,time=int(fn),int(time)
            
            if typ=='start':
                if stack:
                    ans[stack[-1]]+=time-prev_time
                stack.append(fn)
                prev_time=time
            else:
                ans[stack.pop()]+=time-prev_time+1
                prev_time=1+time
        return ans
n=2

logs = ["0:start:0",
 "1:start:2",
 "1:end:5",
 "0:end:6"]
if __name__ == "__main__":
    print(Solution().exclusiveTime( n,logs)  ) 
            
#638. Shopping Offers            
class Solution:
    def shoppingOffers(self, price, special, needs):
        """
        :type price: List[int]
        :type special: List[List[int]]
        :type needs: List[int]
        :rtype: int
        """
       
        def dfs( needs,special,price,currentprice,specialindex):
            if needs==[0 for _ in range(len(needs))]:
               self.res=min(self.res,currentprice)
               return 
           
            for x in range(specialindex,len(special)):
                isspecial=special[x]
                valid=True
                for y in range(len(needs)):
                    if needs[y]< isspecial[y]:
                        valid=False
                        break
                if valid:
                   nextneeds=[needs[a]-isspecial[a]  for a in range(len(needs))]
                   dfs(nextneeds,special,price,currentprice+isspecial[-1],x)
                
            for z in range(len(needs)):
                    if needs[z]>=1:
                       currentprice+=price[z]*needs[z]
            self.res=min(self.res,currentprice)
        self.res=float('inf')
        dfs( needs,special,price,0,0)
        return self.res
if __name__ == "__main__":
    print(Solution().shoppingOffers( [2,5], [[3,0,5],[1,2,10]], [3,2])  )    

#640. Solve the Equation             
class Solution:
    def solveEquation(self, equation):
        """
        :type equation: str
        :rtype: str
        """
        from itertools import groupby   
        left=True
        A=B=0
        sign=1
           
        for k,v in groupby(equation, key= lambda x : x in '+-='):
            
            w = "".join(v)
            print(w)
            if k:
                #for x in w:
                   x=w
                   if x=='=':
                      left=False
                   sign=1 if x!='-' else -1
                   sign*=1 if left else -1
            else:
                if w[-1]=='x':
                    A+=sign*(int(w[:-1] if w[:-1] else 1))
                else:
                    B+=sign* int(w)
        
        if A==B and A==0:
            return "Infinite solutions"
        elif A ==0 :
             return "No solution"
        else:
            return 'x={}'.format(int(-B/A))
equation="1-x+x-x+x+x=-99-2x+x-x+x"     
        
if __name__ == "__main__":
    print(Solution().solveEquation(equation)  )        
        
#646. Maximum Length of Pair Chain        
class Solution:
    def findLongestChain(self, pairs):
        """
        :type pairs: List[List[int]]
        :rtype: int
        """
        pairs.sort(key=lambda x: x[1]) 
        
        curr=float('-inf')
        res=0
        for p in pairs:
            if curr<p[0]:
                curr=p[1]
                res+=1
        return res
if __name__ == "__main__":
    print(Solution().findLongestChain([[1,2], [2,3], [3,4]]  ))            
        
#647. Palindromic Substrings        
class Solution:
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        res=0
        
        if not s:
            return 0
        n=len(s)
        
        dp=[[0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n-1,-1,-1):
            for j in range(i,n):
                if s[i]==s[j] and ( j-i<3 or  dp[i+1][j-1]):
                    
                   dp[i][j]=True
                
                   res+=1
        return res
if __name__ == "__main__":
    print(Solution().countSubstrings('aaa' ))                 
                
#648. Replace Words                   
class Solution:
    def replaceWords(self, dict, sentence):
        """
        :type dict: List[str]
        :type sentence: str
        :rtype: str
        """
        dictset=set(dict)
        
        def replace(word):
            for i in range(1,len(word)):
                if word[:i] in dictset:
                    return word[:i]
            return word
            
        return  ' '.join([replace(word) for word  in  sentence.split() ] )    
            
dict=["cat", "bat", "rat"]
sentence="the cattle was rattled by the battery"            
if __name__ == "__main__":
    print(Solution().replaceWords(dict, sentence ))           
        
        
#649. Dota2 Senate        
class Solution:
    def predictPartyVictory(self, senate):
        """
        :type senate: str
        :rtype: str
        """
        from collections import deque
        q=deque()
        people=[0,0]
        
        ban=[0,0]
        
        for man in senate:
            y=  (man=='R')
            people[y]+=1
            q.append(y)
            
        
        
        while all ( people):
              x=q.popleft()
              if ban[x]:
                  ban[x]-=1
                  people[x]-=1
              else:
                  ban[x^1]+=1
                  q.append(x)
        return "Radiant"  if people[1]  else "Dire"
senate ="RD"
if __name__ == "__main__":
    print(Solution().predictPartyVictory(senate ))            
            
#650. 2 Keys Keyboard            
class Solution:
    def minSteps(self, n):
        """
        :type n: int
        :rtype: int
        """            
        d=2
        ans=0
        while n>1:
            
            while n%d==0:
                  ans+=d
                  n/=d
            d+=1
        return ans
 
           
if __name__ == "__main__":
    print(Solution().minSteps(3 ))         
#652. Find Duplicate Subtrees        
class Solution:
    def findDuplicateSubtrees(self, root):
        """
        :type root: TreeNode
        :rtype: List[TreeNode]
        """
        
        from collections import defaultdict,deque
        
        
        trees=defaultdict(list)
        treeid=defaultdict()
        treeid.default_factory=treeid.__len__
        
        
        def getId(root):
            if root:
               rootid=treeid[root.val,getId(root.left),getId(root.right)]
               trees[rootid].append(root)
               return rootid
        
        getId(root)
        
        return [ node[0] for node in trees.values() if node[1:] ]
        
root=TreeNode(1)      
 
root.left= TreeNode(2) 
root.right= TreeNode(3) 

root.left.left= TreeNode(4)                   
root.right.left= TreeNode(2) 
root.right.right= TreeNode(4) 
         
root.right.left.left= TreeNode(4)        
        
if __name__ == "__main__":
    printT(Solution().findDuplicateSubtrees( root)  ) 

#654. Maximum Binary Tree      
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
#        
#        Input: [3,2,1,6,0,5]
#Output: return the tree root node representing the following tree:
#
#      6
#    /   \
#   3     5
#    \    / 
#     2  0   
#       \
#        1
        
        def build(nums):
            if nums:
                maxn=max(nums)
                ix=nums.index(maxn)
                root=TreeNode(maxn)
                root.left= build(nums[:ix])
                root.right= build(nums[ix+1:])
                return root
        return build(nums)
        
if __name__ == "__main__":
    printT(Solution().constructMaximumBinaryTree([3,2,1,6,0,5])  ) 
        
#655. Print Binary Tree        
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def printTree(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[str]]
        """
        if not root:
            return ['']
        
        def height(root):
            if not root:
                return 0
            return 1+max(height(root.left),height(root.right))
        
        def update(root,row,left,right):
            if not root:
                return 
            mid=(left+right)//2
            
            self.res[row][mid]=str(root.val)
            update(root.left,row+1,left,mid-1)
            update(root.right,row+1,mid+1,right)
            
        high=height(root)
        width=2**high-1
            
        self.res=[['' for _ in range(width)] for _ in range(high)]
        update(root,0,0,width-1)
        
        return self.res
    
root=TreeNode(1)      
 
root.left= TreeNode(2) 
root.right= TreeNode(3) 
#
#root.left.left= TreeNode(4) 
root.left.right= TreeNode(4)                   
#root.right.left= TreeNode(2) 
#root.right.right= TreeNode(4) 
         
#root.right.left.left= TreeNode(4)  
if __name__ == "__main__":
    print(Solution().printTree(root)  )         
        
#658. Find K Closest Elements        
class Solution:
    def findClosestElements(self, arr, k, x):
        """
        :type arr: List[int]
        :type k: int
        :type x: int
        :rtype: List[int]
        """        
#Input: arr=[1,2,3,4,5], k=4, x=3
#Output: [1,2,3,4]       
#Input: [1,2,3,4,5], k=4, x=-1
#Output: [1,2,3,4]        
#        

        if not arr:
            return []
        
        n=len(arr)
        
        if k==n:
            return arr
        
        if k>n:
            return []
        import bisect
        idx=bisect.bisect_left(arr,x)
        
        res=[]
        
        
        idxup=idx
        idxdown=idx-1
        
        
        while k>0:
          #print(k,idxup,idxdown)
          
          while idxdown>=0 and idxup<n and k>0:
             if x-arr[idxdown] <=arr[idxup]-x:
                 res.append(arr[idxdown])
                 idxdown-=1
                 k-=1
             else:
                 res.append(arr[idxup])
                 idxup+=1
                 k-=1
          while idxdown>=0 and k>0:
               res.append(arr[idxdown])
               idxdown-=1
               k-=1
              
          while idxup<n and k>0:
               res.append(arr[idxup])
               idxup+=1
               k-=1   
              
        res.sort()   
        return res
arr=[0,1,1,1,2,3,6,7,8,9]
k=9
x=4
arr=[1,2,3,4,5]
k=4
x=3
arr=[1,2,3,4,5]
k=4
x=-1

arr=[0,0,0,1,3,5,6,7,8,8]
k=2
x=2

if __name__ == "__main__":
    print(Solution().findClosestElements(arr, k, x)  )          
        
        
#659. Split Array into Consecutive Subsequences        
class Solution:
    def isPossible(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        from collections import Counter
        
        count=Counter(nums)
        
        tail=Counter()
        
        for x  in nums:
            if count[x]==0:
                continue
            elif tail[x]>0:
                 tail[x]-=1
                 tail[x+1]+=1
            elif count[x+1]>0 and count[x+2]>0:
                 count[x+1]-=1
                 count[x+2]-=1
                 tail[x+3]+=1
            else:
                return False
            count[x]-=1
        return True
            
if __name__ == "__main__":
    print(Solution().isPossible([1,2,3,3,4,5])  )          
                   
#662. Maximum Width of Binary Tree            
class Solution:
    def widthOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        queue=[(root,0,0)]       
        cur_depth=0
        ans=0
        pos=0
        left=0
        
        for node , depth,pos in queue:
            if node:
                queue.append((node.left,depth+1,pos*2))
                queue.append((node.right,depth+1,pos*2+1))
                if cur_depth!=depth:
                   cur_depth= depth
                   left=pos
                ans=max(-left+pos+1,ans)
        
        return ans
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
root=TreeNode(1)      
 
root.left= TreeNode(3) 
root.right= TreeNode(2) 
#
root.left.left= TreeNode(5) 
root.left.right= TreeNode(3)                   
#root.right.left= TreeNode(2) 
root.right.right= TreeNode(9) 
         
#root.right.left.left= TreeNode(4)  
if __name__ == "__main__":
    print(Solution().widthOfBinaryTree(root)  )                  
            
#667. Beautiful Arrangement II            
class Solution:
    def constructArray(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[int]
        """   
        a=[i for i in range(1,n+1)]         
        for i in range(1,k):
            a[i:]=a[:i-1:-1]
        
        
            
        return a
if __name__ == "__main__":
    print(Solution().constructArray(3,2)  )                  
              
#670. Maximum Swap                
class Solution:
    def maximumSwap(self, num):
        """
        :type num: int
        :rtype: int
        """
        if num <12:
            return num
        nums=list(str(num))
        
        from collections import defaultdict
        bucket=defaultdict(int)
        
        for i,v in enumerate(nums):
            bucket[v]=i
        
        print(bucket)
        for j in range(len(nums)):
            for k in range(9,int(nums[j]),-1):
                if bucket[str(k)]>j:
                   nums[j] , nums[bucket[str(k)]] = nums[bucket[str(k)]], nums[j]
                   s=''.join(nums)
                   return int(s)
        return num
if __name__ == "__main__":
    print(Solution().maximumSwap(9973)  )                    
        
#672. Bulb Switcher II        
class Solution:
    def flipLights(self, n, m):
        """
        :type n: int
        :type m: int
        :rtype: int
        """
        if m==0 :
            return 1
        if n==1 :
            return 2
        if n==2 and m==1:
            return 3
        if n==2:
            return 4
        if m==1:
            return 4
        if m==2 :
            return 7
        if m>=3:
            return 8
        return 8;                
        
#673. Number of Longest Increasing Subsequence        
class Solution:
    def findNumberOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        
        n=len(nums)
        
        
        maxval=1
        count=1
        
        dp1=[1 for _ in range(n)]
        dp2=[1 for _ in range(n)]
        
        for i  in range(1,n):
            for j in range(i):
                if nums[i]> nums[j]:
                    if dp1[j]+1>dp1[i]:
                       dp2[i]=dp2[j]
                       dp1[i]=dp1[j]+1 
                    elif dp1[j]+1==dp1[i]:
                         dp2[i]+= dp2[j]
            if dp1[i]>maxval:
                    count=dp2[i]
                    maxval=  dp1[i]
            elif dp1[i]==maxval:
                      count+=dp2[i]    
                
        return count
if __name__ == "__main__":
    print(Solution().findNumberOfLIS([2,2,2,2,2])  )                    
                                
#676. Implement Magic Dictionary                        
class MagicDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        from collections import defaultdict
        self.worddict=defaultdict(list)

    def buildDict(self, dict):
        """
        Build a dictionary through a list of words
        :type dict: List[str]
        :rtype: void
        """
        for word in dict:
            self.worddict[len( word )]+=[word]

    def search(self, word):
        """
        Returns if there is any word in the trie that equals to the given word after modifying exactly one character
        :type word: str
        :rtype: bool
        """
        
        for w in self.worddict[len( word)]:
            diff=0
            for j in range(len(w)):
                if w[j]!=word[j]:
                   diff+=1
                if diff >1 :
                    break
            if diff==1:
                return True
        return False
 

# Your MagicDictionary object will be instantiated and called as such:
obj = MagicDictionary()
obj.buildDict(["hello", "leetcode"])
obj.search("leetcoded")                        
                        
                        
#677. Map Sum Pairs                       
class MapSum:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        from collections import defaultdict
        self.worddict=defaultdict()

    def insert(self, key, val):
        """
        :type key: str
        :type val: int
        :rtype: void
        """
        self.worddict[key]=val
        

    def sum(self, prefix):
        """
        :type prefix: str
        :rtype: int
        """
        res=0
        
        if not prefix:
            return res
        n=len(prefix)
        for word  in  self.worddict:
            if len(word)>=n:
                if prefix==word[:n]:
                    res+=self.worddict[word]
        return res
        

obj = MapSum()
obj.insert("apple", 3)
obj.sum('ap')  
obj.insert("app", 2)                      
                        
#678. Valid Parenthesis String                        
class Solution:
    def checkValidString(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if not s:
            return True
        
        high=0
        low=0
        
        for string in s:
            if string=='(':
                high+=1
                low+=1
            if string==')':
                high-=1
                if low>0:
                    low-=1
            if string=='*':
                high+=1
                if low>0:
                    low-=1
            if high <0:
                return False
        if low==0:
            return True
        else:
            return False
if __name__ == "__main__":
    print(Solution().checkValidString("(*))")  )             
            
        
#684. Redundant Connection        
class Solution:
    def findRedundantConnection(self, edges):
        """
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        
    
        
        
        n=len(edges)   
        if n==0:
           return []
        if n<3:
            return []
        
        parent=list(range(2*n))
        
        def find(x):
            if x !=parent[x]:
                parent[x]=find(parent[x])
            return parent[x]
        for edge in edges:
            f,t=edge
            if find(f)==find(t):
                return edge
            parent[find(f)]=find(t)
if __name__ == "__main__":
    print(Solution().findRedundantConnection([[3,4],[1,2],[2,4],[3,5],[2,5]])  )              
            
#688. Knight Probability in Chessboard           
class Solution:
    def knightProbability(self, N, K, r, c):
        """
        :type N: int
        :type K: int
        :type r: int
        :type c: int
        :rtype: float
        """
        
        directions=[(1,2),(1,-2),(-1,2),(-1,-2),(2,-1),(2,1),(-2,1),(-2,-1)]  

        dp=[[[0 for _ in range(N)]  for _ in range(N)] for _ in range(K+1) ]

        dp[0][r][c] =1
        for step in range(1,K+1) :
           for i in range(N):
               for j in range(N):
                   for move in directions:
                       x=i+move[0]
                       y=j+move[1]
                       if x<0 or y <0 or x>N-1 or y >N-1:
                           continue
                       dp[step][i][j]+=0.125*  dp[step-1][x][y]
        res=0
        
        for i in range(N):
            for j in range(N):
                res+=dp[K][i][j]
        return res
    
if __name__ == "__main__":
    print(Solution().knightProbability(3, 2, 0, 0)  )            
            
            
#692. Top K Frequent Words            
class Solution:
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        from collections import Counter,defaultdict
        dictw=Counter(words)  
        freqs=defaultdict(list)
        

        for word,count  in dictw.items():
            freqs[count]+=[word]
        
        
        res=[]
        for i in range(len(words)-1,-1,-1):
            if i in freqs:
                for w in freqs[i]:
                   res.append((w,i))
            if len(res)>k:
                break
        
        print(res)
        res.sort(key=lambda a:(-a[1],a[0]))
        
        return [en[0] for en in res[:k]]
words= ["i", "love", "leetcode", "i", "love", "coding"]
k = 2           
if __name__ == "__main__":
    print(Solution().topKFrequent(words, k)   )          
            
#698. Partition to K Equal Sum Subsets            
class Solution:
    def canPartitionKSubsets(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        n=len(nums)  
        if n<k:
            return False
        total=sum(nums)
        if total % k != 0:
            
            return False
        if k==1:
            return True
        
        self.target=sum(nums)/k
        self.visit=[False for _ in range(n)]
        
        def search(k,ind,sumn,cnt):
            if k==1:
                return True
            if self.target==sumn and cnt>0:
                return search(k-1,0,0,0)
            for i in range(ind,n):
                if not self.visit[i]  and nums[i]+sumn<=self.target:
                    self.visit[i]=True
                    if search(k,i+1,nums[i]+sumn,cnt+1):
                        return True
                    self.visit[i]=False
            return False
        return search(k,0,0,0)
nums = [4, 3, 2, 3, 5, 2, 1]
k=4
if __name__ == "__main__":
    print(Solution().canPartitionKSubsets(nums, k)   )             
            
#712. Minimum ASCII Delete Sum for Two Strings            
class Solution:
    def minimumDeleteSum(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: int
        """            
# dp[i][j] is the cost for s1.substr(0,i) and s2.substr(0, j).
# Note s1[i], s2[j] not included in the substring.       

        n1=len(s1)
        n2=len(s2)
        
        dp= [[0 for _ in range(n2+1)] for _ in range(n1+1)]
        
        for i in range(1,n1+1):
            dp[i][0]=dp[i-1][0]+ord(s1[i-1])
        for j in range(1,n2+1):
            dp[0][j]=dp[0][j-1]+ord(s2[j-1])
            
            
        for i in range(1,n1+1):
            for j in range(1,n2+1):
                
                if s1[i-1]==s2[j-1]:
                    dp[i][j]= dp[i-1][j-1]
                    
                else:
                    dp[i][j]=min(dp[i-1][j]+ord(s1[i-1]),dp[i][j-1]+ord(s2[j-1]))
        return dp[n1][n2]
s1 = "sea"
s2 = "eat" 
s1 = "delete"
s2 = "leet"  
if __name__ == "__main__":
    print(Solution().minimumDeleteSum(s1, s2)  )                   

#713. Subarray Product Less Than K                      
class Solution:
    def numSubarrayProductLessThanK(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """                        
        if k<2:
           return  0
        product=1
        count=0
        n=len(nums)
        i=0
        for j in range(n):   
            product*= nums[j]   
            while i<=j  and product>=k:
                 product/=nums[i]
                 i+=1
            count+=j+1-i
        return count
nums = [10, 5, 2, 6]
k = 100
if __name__ == "__main__":
    print(Solution().numSubarrayProductLessThanK(nums, k)  )                         
                        
#714. Best Time to Buy and Sell Stock with Transaction Fee                        
class Solution:
    def maxProfit(self, prices, fee):
        """
        :type prices: List[int]
        :type fee: int
        :rtype: int
        """
#hold[i] : The maximum profit of holding stock until day i;
#notHold[i] : The maximum profit of not hold stock until day i;
#
#dp transition function:
#For day i, we have two situations:
#
#Hold stock:
#(1) We do nothing on day i: hold[i - 1];
#(2) We buy stock on day i: notHold[i - 1] - prices[i - 1] - fee;
#
#Not hold stock:
#(1) We do nothing on day i: notHold[i - 1];
#(2) We sell stock on day i: hold[i - 1] + prices[i - 1];        
        n=len(prices)
        sold=[0 for _ in range(n)] 
        hold=[0 for _ in range(n)] 
        hold[0]=-prices[0]
        for i in range(1,n):
            sold[i]=max(sold[i-1],hold[i-1]+prices[i]-fee)
            hold[i]=max(hold[i-1],sold[i-1]-prices[i])
            
        return sold[n-1]
prices = [1, 3, 2, 8, 4, 9]
fee = 2
if __name__ == "__main__":
    print(Solution().maxProfit(prices, fee)  )

#718. Maximum Length of Repeated Subarray
class Solution:
    def findLength(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: int
        """
        an=len(A)       
        bn=len(B)
        
        dp=[[0  for _ in range(bn+1)] for _ in range(an+1)]
        
        for i in range(1,an+1):
            for j in range(1,bn+1):
                if A[i-1]==B[j-1]:
                    dp[i][j]=dp[i-1][j-1]+1
        return max(max(row) for row in dp)
A= [1,2,3,2,1]
B= [3,2,1,4,7]
if __name__ == "__main__":
    print(Solution().findLength(A, B)  )


#721. Accounts Merge
class Solution:
    def accountsMerge(self, accounts):
        """
        :type accounts: List[List[str]]
        :rtype: List[List[str]]
        """
        
        def find(a):
            if ds[a]<0:
                return a
            ds[a]=find(ds[a])
            return ds[a]
        def union(a,b):
            a,b=find(a),find(b)
            if a!=b:
                if ds[a]<ds[b]:
                   ds[a]+= ds[b]
                   ds[b]=a
                else:
                   ds[b]+= ds[a]
                   ds[a]=b
                   
                   
        c=0
        ds=[]
        email_to_id={}
        id_to_name={}
        
        
        for account in accounts:
            for email in account[1:]:
                if email not in email_to_id:
                    email_to_id[email]=c
                    id_to_name[c]=account[0]
                    c+=1
                    ds.append(-1)
                union(email_to_id[account[1]],email_to_id[email])
        
        from collections import defaultdict
        
        res=defaultdict(list)
        
        for email, id in email_to_id.items():
            master=find(id)
            res[master]+=[email]
        return [[id_to_name[id]]+sorted(res[id])  for id in res]
    
accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], 
            ["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]    
if __name__ == "__main__":
    print(Solution().accountsMerge(accounts)  )

#722. Remove Comments
class Solution:
    def removeComments(self, source):
        """
        :type source: List[str]
        :rtype: List[str]
        """
        import re
        
       
            
#res2=re.sub('//.*|/\\*(.|\\\)*?\\*/','',"\\".join(res)).split('\\')
        
        
        
        res=re.sub('/\\*(.|\\\)*?\\*/','',"\\".join(source)).split('\\')
        res2=[]
        for s in res:
            if s:
                s2=re.sub('//.*','',s)
                if s2:
                   res2.append(s2)
        
        
        return [x for x in res2 if x]
source=["/*Test program */", "int main()", "{ ", "  // variable declaration ", 
                 "int a, b, c;", "/* This is a test", "   multiline  ", "   comment for ",
                 "   testing */", "a = b + c;", "}"]
source = ["main() {", "/* here is commments", "  // still comments */",
          "   double s = 33;", "   cout << s;", "}"]
source = ["a/*comment", "line", "more_comment*/b"]
if __name__ == "__main__":
    print(Solution().removeComments(source)  )            
 
class Solution:
    def removeComments(self, source):
        """
        :type source: List[str]
        :rtype: List[str]
        """
        in_block=False
        ans=[]
        
        for line in source:
            i=0
            if not in_block:
               newline=[]
            while i<len(line):
                if line[i:i+2]=='/*'  and not in_block:
                   in_block=True
                   i+=1
                elif line[i:i+2]=='*/'  and in_block:
                   in_block=False
                   i+=1 
                elif line[i:i+2]=='//'  and not in_block:
                    break
                elif not in_block:
                    newline.append(line[i])
                    #print(line[i])
                    #print(newline)
                i+=1
            if newline and not in_block:
                ans.append(''.join(newline))
        return ans
if __name__ == "__main__":
    print(Solution().removeComments(source)  )  
                
   
                        
#725. Split Linked List in Parts                        
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def splitListToParts(self, root, k):
        """
        :type root: ListNode
        :type k: int
        :rtype: List[ListNode]
        """
       

        
        length=  0
        node=root
        while node:
            node=node.next
            length+=1
            
        width=length//k
        remainder=length%k
        
        ans=[]
        cur=root
        for i in range(k):
            head=cur
            for j in range(width+(i<remainder)-1):
                if cur:
                    cur=cur.next
            
            if cur:
                cur.next,cur=None,cur.next
            
            ans.append(head)
        return ans
    
       
def printnode(n):
      if not n:
         print('#')
      else:
          print(n.val)
          if n.next:
             printnode(n.next)

           
printnode(root)       
       
root=[]   
k=3              
                        
root=ListNode(1)                        
root.next= ListNode(2)                            
root.next.next= ListNode(3)                          
                    
#729. My Calendar I        
class MyCalendar:

    def __init__(self):
        
        self.Calendar=[]
    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """
        if not self.Calendar:
           self.Calendar.append((start,end))
           return True
        else:
            for timeframe in self.Calendar:
                s,e=timeframe
                
                if not (e<= start or s>=end):
                    return False
            self.Calendar.append((start,end))
            return True
            
        


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)        
     
obj.book(10, 20)         
obj.book(15, 25)        
obj.book(20, 30)   

#731. My Calendar II
class MyCalendarTwo:

    def __init__(self):
        
        self.calendar=[]
        self.overlap=[]
    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """
        for i,j in self.overlap:
            if start<j and end >i:
                return False
        for i,j in self.calendar:
            if start<j and end >i:
               self.overlap.append( (max(i,start),min(end,j)))
        self.calendar.append((start, end))  
            
            
        return True


# Your MyCalendarTwo object will be instantiated and called as such:
# obj = MyCalendarTwo()
# param_1 = obj.book(start,end)
obj.book(10, 20)         
obj.book(50, 60)        
obj.book(10, 40)   
obj.book(5, 15)
obj.book(5, 10)
obj.book(25, 55)

#735. Asteroid Collision
class Solution:
    def asteroidCollision(self, asteroids):
        """
        :type asteroids: List[int]
        :rtype: List[int]
        """
        ans=[]
        for new in asteroids:
            while ans and   new<0<ans[-1]:
                if ans[-1] <-new:
                    ans.pop()
                    continue
                elif ans[-1] == -new:
                    ans.pop()
                break
            else:
                ans.append(new)
        return ans
#        def Collision(asteroids):
#            if not asteroids:
#                return []
#            if len(asteroids)==1:
#                return asteroids
#            
#            for i  in range(1,len(asteroids)):
#                if asteroids[i-1] >0 and asteroids[i] <0:
#                    if abs( asteroids[i-1])==abs(asteroids[i]):
#                        return Collision(asteroids[:i-1]+asteroids[i+1:])
#                       
#                    elif abs( asteroids[i-1])>abs(asteroids[i]):
#                        return Collision(asteroids[:i]+asteroids[i+1:])
#                    else:
#                        return Collision(asteroids[:i-1]+asteroids[i:])
#            return asteroids
#        return Collision(asteroids)
        
asteroids=[5, 10, -5]
asteroids = [8, -8]
asteroids = [10, 2, -5]
asteroids = [-2, -1, 1, 2]

if __name__ == "__main__":
    print(Solution().asteroidCollision(asteroids)  ) 

#738. Monotone Increasing Digits
class Solution:
    def monotoneIncreasingDigits(self, N):
        """
        :type N: int
        :rtype: int
        """
        if N<10:
            return N
        i=0
       
        
        listN=list(str(N))
        n=len(listN)
        while i<n-1 and listN[i]<=listN[i+1]:
            i+=1
        
        if i==n-1:
            return N
        
        while i>0 and listN[i]==listN[i-1]:
            i-=1
        
        listN[i]=str(int(listN[i])-1)
        listN[i+1:]=['9' for _ in range(n-i-1)]
        
        return int(''.join(listN))
N=1234
N=333222    
if __name__ == "__main__":
    print(Solution().monotoneIncreasingDigits(N)  ) 
       
        
#739. Daily Temperatures       
class Solution:
    def dailyTemperatures(self, temperatures):
        """
        :type temperatures: List[int]
        :rtype: List[int]
        """        
        ans=[0 for _ in range(len(temperatures))]
        stack=[]
        
        for i in range(len(temperatures)-1,-1,-1):
            while stack and temperatures[i]>=temperatures[stack[-1]]:
                stack.pop()
            if stack:
                ans[i]=stack[-1]-i
            stack.append(i)
        return ans
temperatures=[73, 74, 75, 71, 69, 72, 76, 73]
if __name__ == "__main__":
    print(Solution().dailyTemperatures(temperatures)  )         
        
#740. Delete and Earn        
class Solution:
    def deleteAndEarn(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """ 
        
        result=[0 for _ in range(10001)]
        for num in nums:
            result[num]+=num
        take=0
        skip=0
        for i in range(len(result)):
            takei=skip+result[i]
            skipi=max(skip,take)
            
            take=takei
            skip=skipi
        return max(skip,take)
nums=[2, 2, 3, 3, 3, 4]
if __name__ == "__main__":
    print(Solution().deleteAndEarn(nums)  )          
        
#743. Network Delay Time
class Solution:
    def networkDelayTime(self, times, N, K):
        """
        :type times: List[List[int]]
        :type N: int
        :type K: int
        :rtype: int
        """
        import heapq
        pq=[]
        adj=[[] for _ in range(N+1)]
        
        
        for time in times:
            adj[time[0]].append((time[1],time[2]))
            
        heapq.heappush(pq,(0,K)) 
        
        visited=set()
        res=0
        
        while len(pq) and len(visited)!=N:
            cur=heapq.heappop(pq)
            visited.add(cur[1])
            res=cur[0]
            
            for  node,t  in adj[cur[1]]:
                if node in visited:
                    continue
                heapq.heappush(pq,(t+cur[0],node))
        return res if len(visited)==N else -1


times=[[2,1,1],[2,3,1],[3,4,1]]
N=4
K=2
if __name__ == "__main__":
    print(Solution().networkDelayTime(times, N, K)  )          
        
#748. Shortest Completing Word
class Solution:
    def shortestCompletingWord(self, licensePlate, words):
        """
        :type licensePlate: str
        :type words: List[str]
        :rtype: str
        """
        from collections import defaultdict,Counter
        lpdict=defaultdict(int)
        
        for p in licensePlate:
            if p.isalpha():
                lpdict[p.lower()]+=1
        def isinside(lpdict,worddict):
            
            for key in lpdict:
                if key not in worddict:
                    return False
                if lpdict[key]>worddict[key]:
                    return False
            return True
        
        words.sort(key=lambda x:len(x))
        
        for word in words:
            worddict=Counter(word)
            if isinside(lpdict,worddict):
                return word
   
        
licensePlate = "1s3 PSt"       
words=  ["step", "steps", "stripe", "stepple"]    
        
if __name__ == "__main__":
    print(Solution().shortestCompletingWord(licensePlate, words)  )        
#752. Open the Lock        
class Solution:
    def openLock(self, deadends, target):
        """
        :type deadends: List[str]
        :type target: str
        :rtype: int
        """        
        from collections import deque
        
        q=deque(['0000','marker'])
        visited=set()
        depth=0
        
        # is use deadends as a list , will get time limit exceed
        #list O(N)
        #set O(1)
        deadends=set(deadends)
        
        def successors(node):
            res=[]
            
            for i in range(len(node)):
                num=int(node[i])
                res+=[node[:i]+str((num+1)%10 )+ node[i+1:]   ]
                res+=[node[:i]+str((num-1)%10 )+ node[i+1:]    ]
            return res
        
        while q:
            node=q.popleft()
            if node==target:
                return depth
            if node in visited or node in deadends:
                continue
            if node=='marker' and not q:
                return -1
            if node=='marker':
                q.append('marker')
                depth+=1
            else:
                visited.add(node)
                q.extend(successors(node))
                
        return -1
deadends = ["0201","0101","0102","1212","2002"]
target = "0202" 

deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"]
target = "8888"   

deadends = ["8888"]
target = "0009"
if __name__ == "__main__":
    print(Solution().openLock(deadends, target)  )         
            
#754. Reach a Number            
class Solution:
    def reachNumber(self, target):
        """
        :type target: int
        :rtype: int
        """
        
        import math
        t=abs(target)
        n=  math.floor((2*t)**0.5   ) 


        while True:
            diff=(1+n)*n/2-t
            
            if diff>=0:
                if diff%2==0:
                    return int(n)
            n+=1
target=2
if __name__ == "__main__":
    print(Solution().reachNumber( -12)  )         
                        
#756. Pyramid Transition Matrix                    
class Solution:
    def pyramidTransition(self, bottom, allowed):
        """
        :type bottom: str
        :type allowed: List[str]
        :rtype: bool
        """            
        from collections import defaultdict
        from itertools  import product
        
        f=defaultdict(lambda:defaultdict(list))
        
        
        for a,b,c in allowed:
            f[a][b].append(c)
        
        
        def pyramid(bottom):
            if len(bottom)==1:
                return True
            for i in product(*(f[a][b]  for a , b in zip(bottom[:-1],bottom[1:]))):
                if pyramid(i):
                    return True
            return False
        return  pyramid(bottom)
bottom = "XXYX"
allowed =["XXX", "XXY", "XYX", "XYY", "YXZ"]

if __name__ == "__main__":
    print(Solution().pyramidTransition( bottom, allowed  ) )      
        
        
        
#763. Partition Labels       
class Solution:
    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """        
        n=len(S)
        if n==0:
            return [0]
        if n==1:
            return [1]
        table=[0 for _ in range(26)]
        
        for x in S:
            table[ord(x)-ord('a')]+=1
        i=0
        j=0
        hashset=set()
        counter=0
        
        res=[]
        while j<n:
            c=S[j]
            
            if c not in hashset:
                hashset.add(c)
                counter+=1
            table[ord(c)-ord('a')]-=1
            
            if table[ord(c)-ord('a')]==0:
                counter-=1
                hashset.remove(c)
                
            j+=1
            if counter==0:
               res.append(j-i)
               i=j
        return res
    
S = "ababcbacadefegdehijhklij"        
        
if __name__ == "__main__":
    print(Solution().partitionLabels(S  ) )      
                
#764. Largest Plus Sign       
class Solution:
    def orderOfLargestPlusSign(self, N, mines):
        """
        :type N: int
        :type mines: List[List[int]]
        :rtype: int
        """        
        grid=[[N for _ in range(N)]  for _ in range(N)]
        for mine in mines:
            x,y=mine
            grid[x][y]=0
        
        for i in range(N):
            l,r,u,d=0,0,0,0
            for j in range(N):
                if grid[i][j]==0:
                    l=0
                else:
                    l+=1
                grid[i][j]=min(grid[i][j],l)
            
            for k in range(N-1,-1,-1):
                if grid[i][k]==0:
                    r=0
                else:
                    r+=1
                grid[i][k]=min(grid[i][k],r)    
        
        
            for j in range(N):
                if grid[j][i]==0:
                    u=0
                else:
                    u+=1
                grid[j][i]=min(grid[j][i],u)   
        
            for k in range(N-1,-1,-1):
                if grid[k][i]==0:
                    d=0
                else:
                    d+=1
                grid[k][i]=min(grid[k][i],d)    
        
        res=0
        print(grid)
        for i in range(N):
            for j in range(N):
                if res<grid[i][j]:
                   res=grid[i][j]
        return res
N = 5
mines = [[4, 2]]        
if __name__ == "__main__":
    print(Solution().orderOfLargestPlusSign(N, mines ) )          
        
        
#767. Reorganize String        
class Solution:
    def reorganizeString(self, S):
        """
        :type S: str
        :rtype: str
        """      
        
        
        a=sorted(sorted(S),key=S.count)
        
        m=len(a)//2
        #print(a)
        a[::2],a[1::2]=a[m:],a[:m]
        #print(a)
        
        return ''.join(a) if a[-1:]!=a[-2:-1] else ''
S='aab'
        
if __name__ == "__main__":
    print(Solution().reorganizeString(S ) )                
        
#769. Max Chunks To Make Sorted        
class Solution:
    def maxChunksToSorted(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        n=len(arr)        
        leftmax=[0 for _ in range(n)]
        rightmin=[0 for _ in range(n)]
        
        
        leftmax[0]=arr[0]
        for i in range(1,n):
            
            leftmax[i]=max(leftmax[i-1],arr[i])
            
        rightmin[n-1]=arr[n-1]
        for j in range(n-2,-1,-1):
            
             rightmin[j]=min( rightmin[j+1],arr[j])
        
        
        res=1
        print(leftmax)
        print(rightmin)
        for k in range(n-1):
            
            if leftmax[k]<=rightmin[k+1]:
                res+=1
        return res
arr=[4,3,2,1,0,5]
#arr=[1,0,2,3,4]
if __name__ == "__main__":
    print(Solution().maxChunksToSorted(arr ) )    
        
#775. Global and Local Inversions        
class Solution:
    def isIdealPermutation(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        n=len(A)
        
        if n==1:
            return True
        
        if n==2:
            return True
        
        
        
        maxleft=A[0]
        
        for i in range(0,n-2):
            maxleft=max(maxleft,A[i])
            if maxleft>A[i+2]:
                return False
        return True
A=[1,0,2]
A = [1,2,0]
A = [0,2,3,1]
if __name__ == "__main__":
    print(Solution().isIdealPermutation(A ) )                
            
            
#777. Swap Adjacent in LR String        
class Solution:
    def canTransform(self, start, end):
        """
        :type start: str
        :type end: str
        :rtype: bool
        """
        s=[(c,i) for i,c in enumerate(start) if c == 'R' or c=='L']        
        e=[(c,i) for i,c in enumerate(end) if c == 'R' or c=='L']
        
        if len(s)!=len(e):
            return False
        
        return all( c1==c2 and ((c1=='L' and i1>=i2) or (c1=='R' and i1<=i2))  
                   for (c1,i1),(c2,i2) in zip(s,e))
start = "RXXLRXRXL"
end = "XRLXXRRLX"        
if __name__ == "__main__":
    print(Solution().canTransform( start, end) )         
        
#779. K-th Symbol in Grammar 
class Solution:
    def kthGrammar(self, N, K):
        """
        :type N: int
        :type K: int
        :rtype: int
        """
        
#        s='0'
#        for _ in range(N) :
#            temp=''
#            for t in s:
#                
#            
#                if t=='0':
#                   temp+='01'
#                else:
#                   temp+='10'
#            s=temp
#        return int(s[K-1])
        
        if N==1:
            return 0
        
        if K%2==0:
            if self.kthGrammar( N-1, K//2)==0:
                return 1
            else:
                return 0
        if K%2==1:
            if self.kthGrammar( N-1, (K+1)//2)==0:
                return 0
            else:
                return 1
        
        
        
N = 30
K = 434991989
N = 4
K = 5
if __name__ == "__main__":
    print(Solution().kthGrammar( N, K) ) 


#781. Rabbits in Forest
class Solution:
    def numRabbits(self, answers):
        """
        :type answers: List[int]
        :rtype: int
        """
        from collections import defaultdict
        dd=defaultdict(int)
        
        sumR=0
        for i in answers:
            if i==0:
                
               sumR+=1
            elif i not in dd:
                dd[i]+=1
                sumR+=i+1
            else:
                dd[i]+=1
                if dd[i]==i+1:
                   dd.pop(i)
        return sumR
answers=   [0,0,1,1,1]
if __name__ == "__main__":
    print(Solution().numRabbits(answers) ) 

#785. Is Graph Bipartite?
class Solution:
    def isBipartite(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: bool
        """
        
        set1=set()
        set2=set()
     
        for index , i in enumerate(graph):
            
            if index not in set1 and index not in set2:
                for j in i:
                    if j in set1:
                        
                        set2.add(index)
                        break
                    else:
                        set1.add(index)
                        break
            if index in set1:
                for j in i:
                    if j in set1:
                        return False
                    set2.add(j)
            if index in set2:
                for j in i:
                    if j in set2:
                        return False
                    set1.add(j)
        print(set1,set2)
        return True
graph=[[1,3], [0,2], [1,3], [0,2]]
graph=[[1,2,3], [0,2], [0,1,3], [0,2]]
graph=[[],[2,4,6],[1,4,8,9],[7,8],[1,2,8,9],[6,9],[1,5,7,8,9],[3,6,9],[2,3,4,6,9],[2,4,5,6,7,8]]
if __name__ == "__main__":
    print(Solution().isBipartite(graph) ) 


#787. Cheapest Flights Within K Stops
class Solution:
    def findCheapestPrice(self, n, flights, src, dst, K):
        """
        :type n: int
        :type flights: List[List[int]]
        :type src: int
        :type dst: int
        :type K: int
        :rtype: int
        """
        from collections import defaultdict
        import heapq
        f=defaultdict(dict)
        for a,b,p in flights:
            f[a][b]=p
        
        
        heap=[(0,src,K+1)]
        
        while heap:
            price,i,k=heapq.heappop(heap)
            if dst==i:
                return price
            if k>0:
                for j in  f[i]:
                    heapq.heappush(heap,(price+f[i][j],j,k-1))
        return -1
n = 3
flights = [[0,1,100],[1,2,100],[0,2,500]]
src = 0
dst = 2
K = 1
n = 3
flights = [[0,1,100],[1,2,100],[0,2,500]]
src = 0
dst = 2
K = 0
if __name__ == "__main__":
    print(Solution().findCheapestPrice(n, flights, src, dst, K) )             










            