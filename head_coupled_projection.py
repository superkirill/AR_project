import numpy as np
import cv2

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import OC_detector

verticies = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )

surfaces = (
    (0,1,2,3),
    (3,2,7,6),
    (6,7,5,4),
    (4,5,1,0),
    (1,5,7,2),
    (4,0,3,6)
    )

colors = (
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (0,1,0),
    (1,1,1),
    (0,1,1),
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (1,0,0),
    (1,1,1),
    (0,1,1),
    )

def Cube():
    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x+=1
            glColor3fv(colors[x])
            glVertex3fv(verticies[vertex])
    glEnd()

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()

def apply_projection(pa, pb, pc, pe, n, f):
    """The function takes four float vectors, pa, pb, pc, pe, which are the screen
        corner positions and the eye position as defined above, plus n and f which
        are the near and far plane distances, identical to those passed to
        gluPerspective or glFrustum.
    """
    # Compute an orthonormal basis for the screen.

    vr = pb - pa
    vu = pc - pa

    vr = vr / np.sum(vr)
    vu = vu / np.sum(vu)
    vn = np.cross(vr, vu)
    
    vn = vn / np.sum(vn)

    # Compute the screen corner vectors.

    va = pa - pe
    vb = pb - pe
    vc = pc - pe

    # Find the distance from the eye to screen plane.

    d = -np.dot(va,vn)

    # Find the extent of the perpendicular projection.

    l = np.dot(vr,va) * n / d
    r = np.dot(vr,vb) * n / d
    b = np.dot(vu,va) * n / d
    t = np.dot(vu,vc) * n / d

    # Load the perpendicular projection.

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glFrustum(l, r, b, t, n, f)

    # Rotate the projection to be non-perpendicular.

    M = np.zeros((16))
    M[0] = vr[0]
    M[4] = vr[1]
    M[ 8] = vr[2]
    M[1] = vu[0]
    M[5] = vu[1]
    M[ 9] = vu[2]
    M[2] = vn[0]
    M[6] = vn[1]
    M[10] = vn[2]

    M[15] = 1.0

    glMultMatrixf(M)

    # Move the apex of the frustum to the origin.

    glTranslatef(-pe[0], -pe[1], -pe[2])

    glMatrixMode(GL_MODELVIEW)

def main():
    pygame.init()

    # OpenGL scene is of size 800*600
    width = 800
    height = 600
    display = (width, height)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    # Video taken by the camera is of size 711*270
    face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    screen_width = 7.09
    screen_height = 5.3

    pa = np.array([-screen_width/2, -screen_height/2, -20.0])
    pb = np.array([screen_width/2, -screen_height/2, -20.0])
    pc = np.array([-screen_width/2, screen_height/2, -20.0])
    pe = np.array([0.0, 0.0, 0.0])
    near = 1.0
    far = 20.0
    x_screen = 0.0
    y_screen = 0.0
    left_eye = right_eye = None
    while 1:
        ret, img = cap.read()
        x_ratio = width / img.shape[0]
        y_ratio = height / img.shape[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if OC_detector.cropEyes(img) is not None:
            _,_,left_eye, right_eye = OC_detector.cropEyes(img)
        if left_eye is not None and right_eye is not None:
            cv2.rectangle(img, (left_eye[0], left_eye[1]), (left_eye[2], left_eye[3]), (0, 0, 255), 2)
            cv2.rectangle(img, (right_eye[0], right_eye[1]), (right_eye[2], right_eye[3]), (0, 0, 255), 2)
            middle_x = ((left_eye[2] + left_eye[0])/2 + (right_eye[2] + right_eye[0])/2)/2
            middle_y = ((left_eye[3] + left_eye[1]) / 2 + (right_eye[3] + right_eye[1])/2)/2
            x_screen = (middle_x - img.shape[0] / 2) * x_ratio
            y_screen = (middle_y - img.shape[1] / 2) * y_ratio
            pe = np.array([-x_screen * 0.00125, -y_screen * 0.00166, 0.0])
            print(x_screen / x_ratio, y_screen / y_ratio)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        apply_projection(pa, pb, pc, pe, near, far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -13.0)

        cv2.imshow('img', img)
        Cube()
        pygame.display.flip()

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

