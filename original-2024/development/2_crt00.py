import math

def gcd(p, q):
    if q == 0:
        return p, 1, 0
    d, a, b = gcd(q, p % q)
    return d, b, a - (p // q) * b

def inverse(k, n):
    d, a, _ = gcd(k, n)
    if d > 1:
        print("Inverse does not exist.")
        return 0
    if a > 0:
        return a
    return n + a

def main():
    try:
        with open("out.bmp", "rb") as in1, open("out1.bmp", "rb") as in2, open("out2.bmp", "rb") as in3, open("dec.bmp", "wb") as out:
            A1 = list(in1.read())
            len1 = len(A1)
            print(len1)

            A2 = list(in2.read())
            len2 = len(A2)
            print(len2)

            A3 = list(in3.read())
            len3 = len(A3)
            print(len3)

            m1 = 255 // 3
            m2 = 255 // 5
            m3 = 255 // 17
            w1 = (inverse(m1, 3) * m1) % 255
            w2 = (inverse(m2, 5) * m2) % 255
            w3 = (inverse(m3, 17) * m3) % 255

            Y = []
            for i in range(len1):
                if i < 500:
                    Y.append(A1[i])
                else:
                    Y.append((A1[i] * w1 + A2[i] * w2 + A3[i] * w3) % 255)

            out.write(bytes(Y))
    except Exception as er:
        print(er)

if __name__ == "__main__":
    main()
