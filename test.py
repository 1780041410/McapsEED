import numpy as np

import tensorflow as tf

gate_embedding_ori = tf.get_variable(name='gate_ori',
                                          shape=[10, 1],
                                          initializer=tf.random_uniform_initializer(minval=0,
                                                                                    maxval=1))
gate_embedding = tf.nn.sigmoid(gate_embedding_ori, name='gate_sigmoid')

tt=tf.nn.embedding_lookup(gate_embedding,np.asarray([[1],[2]]))
tt=tf.reshape(tt,[-1,1])
aa=tf.constant(np.asarray([[1,1,1],[2,2,2]],dtype=np.float32))
tt=aa*tt
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(gate_embedding))
    print('*'*100)
    print(sess.run(tt))
    print(tt)

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


'''
def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
    
    head = ListNode(0)
    tail = head
    x = 0
    #x当做进位
    while l1 != None and l2 != None:
        #如果都为None就结束
        l1 = ListNode(0) if l1 == None else l1
        l2 = ListNode(0) if l2 == None else l2
        #如果为None就添加一个0节点
        x, y = divmod(l1.val + l2.val + x, 10)
        tail.next = ListNode(y)
        tail = tail.next
        l1, l2 = l1.next, l2.next
    return head.next

'''
'''
print(divmod(10, 6))
l1=ListNode(2)
l1_1=l1
l1.next=ListNode(4)
l1=l1.next
l1.next=ListNode(3)

l2=ListNode(5)
l2_1=l2
l2.next=ListNode(6)
l2=l2.next
l2.next=ListNode(4)
solution=Solution()
solution.addTwoNumbers(l1_1,l2_1)
class Solution:
    def addTwoNumbers(self, l1, l2):
        head = ListNode(0)
        tail = head
        carry = 0
        # x当做进位
        while l1 or l2:
            # 如果都为None就结束
            x=l1.val if l1  else 0
            y=l2.val if l2  else 0
            # 如果为None就添加一个0节点
            carry,t = divmod(x+y+carry, 10)
            tail.next = ListNode(t)
            tail = tail.next
            l1, l2 = l1.next, l2.next
        while head:
            print(head.val)
            head=head.next
        return head


class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows==1:
            return s
        s_len = len(s)
        c,k = numRows-1,(2*numRows-2)
        arr =["" for i in range(numRows)]
        for i,char in enumerate(s):
            i_0=i%k
            y = i_0 if i_0 < numRows else c - (i_0 -c)
            arr[y] += char
        return "".join(arr)

'''
# def convert(s,numRows):
#     if numRows==1:
#         return s
#     s_len=len(s)
#
#     c,k=numRows-1,(2*numRows-2)
#     print(c,k)
#     arr=['' for i in range(numRows)]
#     for i,char in enumerate(s):
#         i_0=i%k
#         y=i_0 if i_0<numRows else c-(i_0-c)
#         arr[y]+=char
#     return ''.join(arr)
#
#
# s = "LCIRETOESIIGEDHN"
# numRows = 3
# print(convert(s, numRows))








# class Solution:
#     def lengthOfLongestSubstring(self, s):
#         """
#         :type s: str
#         :rtype: int
#         """
#         r=""   #储存无重复子串
#         maxNum=0
#         for i in s:
#             if i not in r:           #如果不在子串里，就代表无重复，直接插进去
#                 r=r+i
#             else:                     #如果在子串里，就代表重复了，不能直接插进去
#                 if len(r)>maxNum:maxNum=len(r)     #先算出来上一个子串的长度
#                 r = r[r.index(i)+1:]+i
#                 print(r)#把这个相同字符后面的保留。比如"dvdf"。第一个子串是"dv",再遍历到d的时候，需要把第一个d后面的v保留，再形成一个"vd"子串,这样还是无重复子串。不保留v的话，就不是一个完整的无重复子串了
#         if len(r) > maxNum: maxNum = len(r)
#         return maxNum

# s=""
# a=Solution()
# print(a.lengthOfLongestSubstring(s))

# def findMedianSortedArrays( nums1, nums2) -> float:
#     nums=nums1+nums2
#     nums=sorted(nums)
#     if len(nums)%2==0:
#         return (nums[len(nums)//2-1]+ nums[len(nums)//2])/2
#     else:
#         return (nums[len(nums)//2])
#
#
#
# print(findMedianSortedArrays([1, 2, 3], [1, 3]))

#
# def longestPalindrome(s):
#     if len(s)<2 or s==s[::-1]:
#         return s
#     start,maxlen=0,1
#     for i in range(len(s)):
#         odd=s[i-maxlen-1:i+1]
#         even=s[i-maxlen:i+1]
#         if i-maxlen-1>=0 and odd==odd[::-1]:
#             start=i-maxlen-1
#             maxlen+=2
#         elif i-maxlen>=0 and even==even[::-1]:
#             start=i-maxlen
#             maxlen+=1
#     return s[start:start+maxlen]








# print(longestPalindrome('babad'))



#
# rel=np.asarray([1,2,3,4,5,6,7,8,9,10,11,12],dtype=np.float32).reshape([2,3,2])
# head=np.asarray([1,2,1,1,1,1,1,1,1,1,1,1],dtype=np.float32).reshape([2,3,2])
# # print(rel)
# rel=tf.constant(rel)
# rel=tf.reshape(rel[:,1,:],[-1,2,1])
# head=tf.constant(head)
# theates=tf.nn.softmax(tf.matmul(head,rel),dim=1)
# theates=head*theates
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
    # print(sess.run(head))
    # print(sess.run(rel))
    # print(theates)

#
# Triple=np.asarray([1,2,3,4,5,6],dtype=np.float32).reshape([2,3])
#
# Triple=tf.constant(Triple)
# head=tf.reshape(Triple[:,0],[-1,1])
# tail=tf.reshape(Triple[:,-1],[-1,1])
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(head))
#     print(sess.run(tail))



'''


	inputs = self.word_embedding(inputs)  # (200,60,200)
		if inputs.shape[0]==rel.shape[0]:
			rel= rel.reshape([-1, self.embed_dim, 1])
			theata = F.softmax(torch.matmul(inputs, rel), dim=1)
			inputs=inputs*theata
		inputs = torch.transpose(inputs, 1, 2)#(200,200,60)
		conv = self.conv_layer(inputs)
		activated = F.tanh(conv)#(200,200,60)

		pooled = torch.sum(activated, 2)#(200,200)
# def fixedPositionEmbedding(batchSize, sequenceLen):
#     embeddedPosition = []
#     for i in range(batchSize):
#         c=[]
#         for j in range(sequenceLen):
#             a=np.zeros(5)
#             a[j]=1
#             c.append(a)
#         embeddedPosition.append(c)
#     return np.asarray(embeddedPosition,dtype=np.float32)
#
#
# aa=fixedPositionEmbedding(2, 5)
# print(aa.shape)
# bb=[
#     [1,2,3],
#     [4,5,6]
# ]
#
# x=np.array([[1,2,3,4],[1,2,3,4]])
# temp=np.exp(x)/np.reshape(np.sum(np.exp(x),axis=1),[x.shape[0],-1])
# print(temp)

#[0.03205860328008499, 0.08714431874203257, 0.23688281808991013, 0.6439142598879722]
'''

#
# head=tf.constant(np.asarray([1,2,3,4],dtype=np.float32).reshape([2,2]))
# rel=tf.constant(np.asarray([5,6,7,8],dtype=np.float32).reshape([2,2]))
# tail=tf.constant(np.asarray([9,10,11,12],dtype=np.float32).reshape([2,2]))
# all=tf.concat([head,rel,tail],axis=-1)
# all=tf.reshape(all,[2,3,2])
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     # print(sess.run(head))
#     print(sess.run(all))
#
