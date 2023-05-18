from cvxpy import *
import numpy as np
#c,g,C,G=Variable(2),Variable(4),Variable((2,2)),Variable((4,4))
#expr1,expr2=np.array([[C,c],[c.T,1]]),np.array([[G,g],[g.T,1]])
#print(c)
#print(expr1)
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
X=np.array([[2,4,5],[3,6,7],[1,8,9]])
print(a.dot(X))
print(a.dot(X).trace())
'''
def UltraSound(self):
    while True:
        GPIO.output(self.Trig, 1)
        time.sleep(0.00001)
        GPIO.output(self.Trig, 0)
        i = 0
        while GPIO.input(self.Echo) == 0:
            i = i + 1
            if i > 1000:
                break
            pass
        time1 = time.time()
        # print(time1)
        while GPIO.input(self.Echo) == 1:
            pass
        time2 = time.time()
        # print(time2)
        duration = time2 - time1
        self.distance = duration * 340 / 2 * 100
        print(self.distance)
'''
a=[1,2,3,4,5]
print(a+10)
print(a[-1])


class Solution:

    def nextPermutation(self, nums: List[int]) -> None:
        end = nums[-1]
        length = len(nums)
        if length == 1:
            nums = nums
        else:
            i = length - 2
            for i in range(length - 1, -1, -1):
                if end > nums[i]:
                    flag = 1
                    nums[-1], nums[i] = nums[i], nums[-1]
                    sort(nums[i + 1:])
                    break
            if flag == 0:
                nextPermutation(nums[:length - 1])
