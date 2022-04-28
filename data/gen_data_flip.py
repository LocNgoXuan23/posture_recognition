import cv2
import matplotlib.pyplot as plt
import os
import shutil


def flipCopyFolderToFolder(m, n, type, data_dir, flip_data_dir):
#   only copy -> type = 0
#   only flip -> type = 1
#   both copy flip -> type = 2
	if type == 0:
		print(f'copy from {m} to {n}...')
	elif type == 1:
		print(f'flip from {m} to {n}')
	elif type == 2:
		print(f'copy flip from {m} to {n}')

	for index, i in enumerate(os.listdir(f'{data_dir}/{str(m)}')):
		img = cv2.imread(f'{data_dir}/{str(m)}/{i}')
		flip_img = cv2.flip(img, 1)
		if type == 0 or type == 2:
			cv2.imwrite(f'{flip_data_dir}/{str(n)}/{i}', img)
		if type == 1 or type == 2:
			cv2.imwrite(f'{flip_data_dir}/{str(n)}/flip_{i}', flip_img)
		

	print('len data_dir : ', len(os.listdir(f'{data_dir}/{str(m)}')))
	print('len flip_data_dir', len(os.listdir(f'{flip_data_dir}/{str(n)}')))


for k in range(1, 6):
	print(f'-------------FOLK {k}----------------')
	data_dir = f'FOLD_{k}/train'
	flip_data_dir = f'FOLD_{k}/train_aug'

	try:
		shutil.rmtree(flip_data_dir)
	except:
		pass
	os.mkdir(flip_data_dir)
	for i in os.listdir(data_dir):
		os.mkdir(flip_data_dir + '/' + str(i))

	for i in range(1, 24):
		flipCopyFolderToFolder(i, i, 0, data_dir, flip_data_dir)
		print('---')

	# flip from 1 to 1
	# flip from 2 to 2
	# flip from 3 to 3
	flipCopyFolderToFolder(1, 1, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(2, 2, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(3, 3, 1, data_dir, flip_data_dir)

	# flip from 4 to 5
	# flip from 5 to 4
	flipCopyFolderToFolder(4, 5, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(5, 4, 1, data_dir, flip_data_dir)

	# flip from 6 to 6
	# flip from 7 to 7
	flipCopyFolderToFolder(6, 6, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(7, 7, 1, data_dir, flip_data_dir)

	# flip from 8 to 9
	# flip from 9 to 8
	flipCopyFolderToFolder(8, 9, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(9, 8, 1, data_dir, flip_data_dir)

	# flip from 10 to 15
	# flip from 11 to 16
	# flip from 12 to 17
	# flip from 13 to 18
	# flip from 14 to 19
	flipCopyFolderToFolder(10, 15, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(11, 16, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(12, 17, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(13, 18, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(14, 19, 1, data_dir, flip_data_dir)

	# flip from 15 to 10
	# flip from 16 to 11
	# flip from 17 to 12
	# flip from 18 to 13
	# flip from 19 to 14
	flipCopyFolderToFolder(15, 10, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(16, 11, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(17, 12, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(18, 13, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(19, 14, 1, data_dir, flip_data_dir)

	# flip from 20 to 20
	# flip from 21 to 22
	# flip from 22 to 21
	# flip from 23 to 23
	flipCopyFolderToFolder(20, 20, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(21, 22, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(22, 21, 1, data_dir, flip_data_dir)
	flipCopyFolderToFolder(23, 23, 1, data_dir, flip_data_dir)







