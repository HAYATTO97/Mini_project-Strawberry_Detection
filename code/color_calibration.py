import cv2
import numpy as np
import os
import glob

# === 이미지 처리 클래스 정의 ===
class ImageProcessor:
    def __init__(self, input_dir, output_dir, apply_grayworld=True, apply_simplewb=True, apply_hsv_boost=True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.apply_grayworld = apply_grayworld
        self.apply_simplewb = apply_simplewb
        self.apply_hsv_boost = apply_hsv_boost
        self.valid_extensions = ('.jpg', '.JPG')

    # 한글 경로 문제 해결
    @staticmethod
    def imread_unicode(path):
        stream = np.fromfile(path, np.uint8)
        return cv2.imdecode(stream, cv2.IMREAD_COLOR)

    @staticmethod
    def imwrite_unicode(path, img):
        ext = os.path.splitext(path)[1]
        success, encoded_img = cv2.imencode(ext, img)
        if success:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            encoded_img.tofile(path)
            return True
        return False

    # === Gray world 변환 ===
    @staticmethod
    def gray_world(img):
        avg_b = np.mean(img[:, :, 0])
        avg_g = np.mean(img[:, :, 1])
        avg_r = np.mean(img[:, :, 2])
        avg = (avg_b + avg_g + avg_r) / 3

        scale_b = avg / avg_b
        scale_g = avg / avg_g
        scale_r = avg / avg_r

        img_balanced = img.astype(np.float32)
        img_balanced[:, :, 0] *= scale_b
        img_balanced[:, :, 1] *= scale_g
        img_balanced[:, :, 2] *= scale_r

        return np.clip(img_balanced, 0, 255).astype(np.uint8)

    # === Simple white balance 변환(OpenCV 내장) ===
    @staticmethod
    def simple_white_balance(img):
        wb = cv2.xphoto.createSimpleWB()
        return wb.balanceWhite(img)

    # === HSV 채널에서 S(채도), V(명도) 히스토그램 평활화 적용 ===
    @staticmethod
    def hsv_brightness_saturation_boost(img): # HSV 채널 중 S, v 채널
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s_eq = cv2.equalizeHist(s)
        v_eq = cv2.equalizeHist(v)
        hsv_eq = cv2.merge([h, s_eq, v_eq])
        return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    # === 모든 이미지에 대해 처리 실행 ===
    def process_all(self):
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if not file.lower().endswith(self.valid_extensions):
                    continue

                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, self.input_dir)
                base, ext = os.path.splitext(rel_path)

                print(f"처리 중: {input_path}")
                img = self.imread_unicode(input_path)
                if img is None:
                    print(f"❌ 이미지 로드 실패: {input_path}")
                    continue

                 # === Gray World 적용 및 저장 ===
                if self.apply_grayworld:
                    out_path = os.path.join(self.output_dir, f"{base}_grayworld{ext}")
                    self.imwrite_unicode(out_path, self.gray_world(img))
                    print(f"✅ Gray World 저장: {out_path}")

                # === Simple White Balance 적용 및 저장 ===
                if self.apply_simplewb:
                    try:
                        out_path = os.path.join(self.output_dir, f"{base}_simplewb{ext}")
                        self.imwrite_unicode(out_path, self.simple_white_balance(img))
                        print(f"✅ Simple WB 저장: {out_path}")
                    except Exception as e:
                        print(f"❌ Simple WB 실패: {input_path}, 에러: {e}")

                # === HSV Boost 적용 및 저장 ===
                if self.apply_hsv_boost:
                    out_path = os.path.join(self.output_dir, f"{base}_hsvboost{ext}")
                    self.imwrite_unicode(out_path, self.hsv_brightness_saturation_boost(img))
                    print(f"✅ HSV Boost 저장: {out_path}")

                print()

# === 실행 ===
if __name__ == "__main__":
    input_dir = r"F:\세종대학교\강의\박사1학기\생명과학을위한머신러닝응용및실습\mini_project\data\strawberry_2w\original\mod\images"  # 처리할 이미지들이 들어있는 폴더
    output_dir = r"F:\세종대학교\강의\박사1학기\생명과학을위한머신러닝응용및실습\mini_project\data\strawberry_2w"                     # 결과를 저장할 상위 폴더

    processor = ImageProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        apply_grayworld=True,
        apply_simplewb=True,
        apply_hsv_boost=True
    )
    processor.process_all()