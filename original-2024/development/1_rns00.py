import numpy as np

def gcd(p, q):
    if q == 0:
        return p, 1, 0
    d, a, b = gcd(q, p % q)
    return d, b, a - (p // q) * b

def inverse(k, n):
    d, a, b = gcd(k, n)
    if d > 1:
        print("Inverse does not exist.")
        return 0
    if a > 0:
        return a
    return n + a

def main():
    with open("lena.bmp", "rb") as in_file:
        a1 = np.fromfile(in_file, dtype=np.uint8)

    len1 = len(a1)
    print(len1)

    B1 = np.zeros(len1, dtype=np.uint8)
    B2 = np.zeros(len1, dtype=np.uint8)
    B3 = np.zeros(len1, dtype=np.uint8)

    for i in range(len1):
        if i < 500:
            B1[i] = a1[i]
            B2[i] = a1[i]
            B3[i] = a1[i]
        else:
            B1[i] = a1[i] % 3
            B2[i] = a1[i] % 5
            B3[i] = a1[i] % 17

    with open("out.bmp", "wb") as out_file, \
         open("out1.bmp", "wb") as out1_file, \
         open("out2.bmp", "wb") as out2_file:

        B1.tofile(out_file)
        B2.tofile(out1_file)
        B3.tofile(out2_file)

if __name__ == "__main__":
    main()
