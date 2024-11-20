# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:20:53 2024

@author: akash
"""

#sorting algorithms
#1 selection sort
#3 insertion sort
#4 quick sort
#5 merge sort




#1 selection sort

def selection_sort(arr):
    n=len(arr)
    
    for i in range(n):
        min_idx=i
        
        for j in range(i+1,n):
            if arr[j]<arr[min_idx]:
                min_idx=j
        arr[i],arr[min_idx]=arr[min_idx],arr[i]
        
    return arr



arr=[1,4,3,2,6,7,4]
print(selection_sort(arr))



#2 insertion sort

def insertion_sort(arr):
    n=len(arr)  
    for i in range(1,n):
        key=arr[i]
        j=i-1
        while j>=0 and arr[j]>key:
            arr[j+1]=arr[j]
            j-=1
        arr[j+1]=key
    return arr
        
        
print(insertion_sort(arr))





#3 quick sort:
    
    
    
def quick_sort(arr):
    n=len(arr)
    if n<=1:
        return arr
    else:
        pivot=arr[n//2]
        left=[x for x in arr if x<pivot]
        mid=[x for x in arr if x==pivot]
        right=[x for x in arr if x>pivot]
        return quick_sort(left)+mid+quick_sort(right)




arr=[3,4,5,7,1,2,4,4]
print(quick_sort(arr))




#4 merge sort


def merge_sort(arr):
    if len(arr)>1:
        mid=len(arr)//2
        left=arr[:mid]
        right=arr[mid:]
        merge_sort(left)
        merge_sort(right)
        #merge
        i=j=k=0
        while i<len(left) and j<len(right):
            if left[i]<right[j]:
                arr[k]=left[i]
                i+=1
                k+=1
            else:
                arr[k]=right[j]
                j+=1
                k+=1
            #k+=1
        while i<len(left):
            arr[k]=left[i]
            i+=1
            k+=1
        while j<len(right):
            arr[k]=right[j]
            j+=1
            k+=1
    return arr



arr=[4,6,2,3,4,9,1]
print(merge_sort(arr))
    


