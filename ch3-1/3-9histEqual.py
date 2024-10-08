import cv2
import matplotlib.pyplot as plt

fig=plt.figure()
rows=3 #열
cols=2 #행

img=cv2.imread('mistyroad.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)		    # 명암 영상으로 변환하고 출력
ax1=fig.add_subplot(rows,cols,1)
ax1.axis("off")
ax1.imshow(gray,cmap='gray')

h=cv2.calcHist([gray],[0],None,[256],[0,256])	    # 히스토그램을 구해 출력
print('histogram', h)
ax2=fig.add_subplot(rows,cols,2)
ax2.plot(h,color='r',linewidth=1)

equal=cv2.equalizeHist(gray)			            # 히스토그램을 평활화하고 출력
ax3=fig.add_subplot(rows,cols,3)
ax3.axis("off")
ax3.imshow(equal,cmap='gray')

h2=cv2.calcHist([equal],[0],None,[256],[0,256])	    # 히스토그램을 구해 출력
print('histogram equal', h)
ax4=fig.add_subplot(rows,cols,4)
ax4.plot(h2,color='r',linewidth=1)



equal2=cv2.equalizeHist(equal)			            # 평활화된걸 또한번 평활화
ax5=fig.add_subplot(rows,cols,5)
ax5.axis("off")
ax5.imshow(equal2,cmap='gray')

h2=cv2.calcHist([equal2],[0],None,[256],[0,256])	    # 히스토그램을 구해 출력
print('histogram equal 2', h2)
ax6=fig.add_subplot(rows,cols,6)
ax6.plot(h2,color='r',linewidth=1)

plt.show()


# import cv2
# import matplotlib.pyplot as plt
#
# fig=plt.figure()
# rows=2
# cols=2
#
# img=cv2.imread('mistyroad.jpg')
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)		    # 명암 영상으로 변환하고 출력
# ax1=fig.add_subplot(rows,cols,1)
# ax1.axis("off")
# ax1.imshow(gray,cmap='gray')
#
# h=cv2.calcHist([gray],[0],None,[256],[0,256])	    # 히스토그램을 구해 출력
# ax2=fig.add_subplot(rows,cols,2)
# ax2.plot(h,color='r',linewidth=1)
#
# equal=cv2.equalizeHist(gray)			            # 히스토그램을 평활화하고 출력
# ax3=fig.add_subplot(rows,cols,3)
# ax3.axis("off")
# ax3.imshow(equal,cmap='gray')
#
# h=cv2.calcHist([equal],[0],None,[256],[0,256])	    # 히스토그램을 구해 출력
# ax4=fig.add_subplot(rows,cols,4)
# ax4.plot(h,color='r',linewidth=1)
#
# plt.show()