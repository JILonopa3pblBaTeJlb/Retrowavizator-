import pygame
import moderngl
import numpy as np
import time
import os
import imageio
import subprocess

# ==========================================
# ШЕЙДЕРЫ (БЕЗ ИЗМЕНЕНИЙ В ЛОГИКЕ)
# ==========================================

BG_VERT = """
#version 330
in vec2 in_vert;
out vec2 v_uv;
void main() {
    v_uv = in_vert;
    gl_Position = vec4(in_vert, 0.999, 1.0);
}
"""

BG_FRAG = """
#version 330
in vec2 v_uv;
out vec4 f_color;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    vec3 color_top = vec3(20.0/255.0, 0.0/255.0, 40.0/255.0);
    vec3 color_bottom = vec3(60.0/255.0, 10.0/255.0, 30.0/255.0);
    float t = clamp((v_uv.y + 0.2) * 0.8, 0.0, 1.0);
    vec3 bg = mix(color_bottom, color_top, t);
    float stars = pow(hash(v_uv * 100.0), 50.0);
    stars += pow(hash(v_uv * 150.0), 80.0);
    f_color = vec4(bg + stars * 0.8, 1.0);
}
"""

SUN_VERT = """
#version 330
in vec2 in_vert;
out float v_y;
out vec2 v_uv;
void main() {
    v_y = in_vert.y;
    v_uv = in_vert;
    gl_Position = vec4(in_vert, 0.998, 1.0);
}
"""

SUN_FRAG = """
#version 330
in float v_y;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 color_top = vec3(255.0/255.0, 160.0/255.0, 0.0/255.0);
    vec3 color_bottom = vec3(255.0/255.0, 20.0/255.0, 120.0/255.0);
    float t = clamp((v_y + 0.05) * 1.8, 0.0, 1.0);
    vec3 color = mix(color_bottom, color_top, t);
    float stripe = sin(v_y * 40.0);
    if (v_y < 0.3 && stripe < -0.4) discard;
    f_color = vec4(color, 1.0);
}
"""

GRID_VERT = """
#version 330
in vec3 in_vert;
uniform mat4 mvp;
uniform float u_offset_z;
uniform float u_loop_dist; 
uniform float u_audio_rms;
uniform vec2 u_blur_offset;
out float v_height;
out vec3 v_local_pos;

#define PI 3.14159265359

float get_h(vec2 p) {
    float road_w = 12.0;
    float d = abs(p.x) - road_w;
    if (d < 0.0) return 0.0;
    
    float rms_factor = 0.3 + u_audio_rms * 0.7; 
    float base_slope = pow(d * 0.05, 1.6);
    float unit = (2.0 * PI) / u_loop_dist;
    
    float h = abs(sin(p.x * 0.15 + p.y * unit)) + abs(cos(p.y * unit * 2.0 - p.x * 0.08));
    h += abs(sin(p.x * 0.3)) * 0.5;
    
    h = pow(h, 2.2); 
    float final_h = h * base_slope * rms_factor * 4.0;
    return min(final_h, 80.0); 
}

void main() {
    vec3 p = in_vert;
    p.y = get_h(vec2(p.x, p.z - u_offset_z));
    v_height = p.y;
    v_local_pos = p;
    vec4 pos = mvp * vec4(p.x, p.y - 5.0, p.z, 1.0);
    pos.xy += u_blur_offset * pos.w;
    gl_Position = pos;
    gl_PointSize = (1.0 - pos.z / pos.w) * 15.0; 
}
"""

FRAG = """
#version 330
in float v_height;
in vec3 v_local_pos;
out vec4 f_color;
uniform vec3 road_color;
uniform vec3 sky_color;
uniform float u_glow;
uniform bool is_obj;
uniform bool is_point;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec3 final_color;
    if (is_obj) {
        float hue = fract(v_local_pos.y * 0.5 + v_local_pos.z * 0.3 + v_local_pos.x * 0.1);
        final_color = hsv2rgb(vec3(hue, 0.8, 1.0));
    } else {
        float t = clamp(v_height * 0.05, 0.0, 1.0);
        final_color = mix(road_color, sky_color, t);
    }
    
    if (is_point) {
        f_color = vec4(final_color * u_glow * 2.0, 1.0);
    } else {
        f_color = vec4(final_color * u_glow, 1.0);
    }
}
"""

OBJ_VERT = """
#version 330
in vec3 in_vert;
uniform mat4 mvp;
uniform vec2 u_blur_offset;
out float v_height;
out vec3 v_local_pos;
void main() {
    v_height = 0.0;
    v_local_pos = in_vert;
    vec4 pos = mvp * vec4(in_vert, 1.0);
    pos.xy += u_blur_offset * pos.w;
    gl_Position = pos;
}
"""

# ==========================================
# ДВИЖОК
# ==========================================

class OBJModel:
    def __init__(self, filename=None):
        self.vertices = []
        self.edges = []
        if filename and os.path.exists(filename):
            self.load_obj(filename)
            self.normalize()
        else:
            self.create_cube()

    def create_cube(self):
        self.vertices = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
        self.edges = [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7]

    def load_obj(self, filename):
        verts = []
        indices = set()
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('v '):
                        verts.append([float(x) for x in line.split()[1:4]])
                    elif line.startswith('f '):
                        p = line.split()[1:]
                        fv = [int(x.split('/')[0]) - 1 for x in p]
                        for i in range(len(fv)):
                            v1, v2 = fv[i], fv[(i + 1) % len(fv)]
                            indices.add(tuple(sorted((v1, v2))))
            self.vertices = verts
            self.edges = [idx for edge in indices for idx in edge]
        except:
            self.create_cube()

    def normalize(self):
        if not self.vertices: return
        v = np.array(self.vertices)
        v -= v.mean(axis=0)
        scale = np.abs(v).max()
        if scale > 0: v /= scale
        self.vertices = v.tolist()

class HQGifMaker:
    def __init__(self):
        pygame.init()
        self.res = (640, 360)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
        pygame.display.set_mode(self.res, pygame.OPENGL | pygame.DOUBLEBUF)
        
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        
        self.loop_duration = 4.0
        self.loop_dist = 40.0
        self.speed = self.loop_dist / self.loop_duration

        self.model_scale = 13.0
        self.obj_pos = [0.0, 1.0, -16.0]
        self.mouse_rot = [0.0, 0.0]
        self.auto_rotate = True
        self.dragging = False

        self.neon_cyan = (0.2, 0.9, 1.0)
        self.neon_pink = (1.0, 0.1, 0.6)

        self.grid_prog = self.ctx.program(vertex_shader=GRID_VERT, fragment_shader=FRAG)
        self.obj_prog = self.ctx.program(vertex_shader=OBJ_VERT, fragment_shader=FRAG)
        self.bg_prog = self.ctx.program(vertex_shader=BG_VERT, fragment_shader=BG_FRAG)
        self.sun_prog = self.ctx.program(vertex_shader=SUN_VERT, fragment_shader=SUN_FRAG)
        
        self.bg_vao = self.create_bg_quad()
        self.sun_vao = self.create_sun_mesh()
        self.grid_vao = self.create_wide_grid()
        
        self.obj_dir = 'obj'
        if not os.path.exists(self.obj_dir): os.makedirs(self.obj_dir)
        self.obj_files = sorted([os.path.join(self.obj_dir, f) for f in os.listdir(self.obj_dir) if f.lower().endswith('.obj')])
        self.obj_index = 0
        self.obj_vao = None
        self.load_current_model()
        
        self.start_t = time.time()
        self.is_recording = False

    def load_current_model(self):
        target = self.obj_files[self.obj_index] if self.obj_files else None
        model = OBJModel(target)
        v_data = np.array(model.vertices, dtype='f4').tobytes()
        e_data = np.array(model.edges, dtype='i4').tobytes()
        if self.obj_vao: self.obj_vao.release()
        self.obj_vao = self.ctx.simple_vertex_array(self.obj_prog, self.ctx.buffer(v_data), 'in_vert', index_buffer=self.ctx.buffer(e_data))

    def create_bg_quad(self):
        return self.ctx.simple_vertex_array(self.bg_prog, self.ctx.buffer(np.array([-1,-0.05, 1,-0.05, -1,1, 1,1], dtype='f4').tobytes()), 'in_vert')

    def create_sun_mesh(self):
        res = []
        for i in range(65):
            theta = np.pi * (i / 64)
            res.append([np.cos(theta) * 0.65 * (360/640), np.sin(theta) * 0.65 - 0.05])
        res = [[0.0, -0.05]] + res
        return self.ctx.simple_vertex_array(self.sun_prog, self.ctx.buffer(np.array(res, dtype='f4').tobytes()), 'in_vert')

    def create_wide_grid(self):
        step = 4.0
        x_range = np.arange(-120, 121, step, dtype='f4')
        z_range = np.arange(-140, 21, step, dtype='f4')
        lines = []
        for z in z_range:
            for i in range(len(x_range)-1): lines.extend([x_range[i],0,z, x_range[i+1],0,z])
        for x in x_range:
            for j in range(len(z_range)-1): lines.extend([x,0,z_range[j], x,0,z_range[j+1]])
        return self.ctx.simple_vertex_array(self.grid_prog, self.ctx.buffer(np.array(lines, dtype='f4').tobytes()), 'in_vert')

    def render_scene(self, t):
        self.ctx.clear(0.02, 0.0, 0.04)
        self.ctx.disable(moderngl.BLEND)
        self.bg_vao.render(moderngl.TRIANGLE_STRIP)
        self.sun_vao.render(moderngl.TRIANGLE_FAN)
        
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ADDITIVE_BLENDING
        
        dist = t * self.speed
        fov, asp = 75, self.res[0]/self.res[1]
        tf = np.tan(np.radians(fov)/2)
        n, f = 0.1, 800.0
        proj = np.array([[1/(tf*asp),0,0,0],[0,1/tf,0,0],[0,0,(f+n)/(n-f),(2*f*n)/(n-f)],[0,0,-1,0]], dtype='f4')
        
        view = np.eye(4, dtype='f4')
        view[1, 3], view[2, 3] = 0.5, -25.0

        gv = view.copy(); gv[2, 3] += (dist % 4.0)
        self.grid_prog['u_offset_z'].value = dist - (dist % 4.0)
        self.grid_prog['u_loop_dist'].value = float(self.loop_dist)
        self.grid_prog['u_audio_rms'].value = 0.0
        self.grid_prog['mvp'].write((proj @ gv).T.copy(order='C').tobytes())
        self.grid_prog['road_color'].value = self.neon_cyan
        self.grid_prog['sky_color'].value = self.neon_pink
        self.grid_prog['is_obj'].value = False
        
        self.grid_prog['is_point'].value = False
        self.render_glow_layer(self.grid_vao, self.grid_prog, 0.4, 0.0018, moderngl.LINES)
        self.grid_prog['is_point'].value = True
        self.grid_vao.render(moderngl.POINTS)

        ax = self.mouse_rot[0]
        if self.auto_rotate:
            ay = (t / self.loop_duration) * (2.0 * np.pi) + self.mouse_rot[1]
        else:
            ay = self.mouse_rot[1]
            
        sx, cx, sy, cy = np.sin(ax), np.cos(ax), np.sin(ay), np.cos(ay)
        rx = np.array([[1,0,0,0],[0,cx,-sx,0],[0,sx,cx,0],[0,0,0,1]], dtype='f4')
        ry = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]], dtype='f4')
        
        scale_mat = np.eye(4, dtype='f4') * self.model_scale; scale_mat[3, 3] = 1.0
        ov = view.copy()
        ov[0, 3], ov[1, 3], ov[2, 3] = self.obj_pos[0], self.obj_pos[1], self.obj_pos[2]
        
        if self.obj_vao:
            self.obj_prog['mvp'].write((proj @ ov @ rx @ ry @ scale_mat).T.copy(order='C').tobytes())
            self.obj_prog['is_obj'].value = True
            self.obj_prog['is_point'].value = False
            self.render_glow_layer(self.obj_vao, self.obj_prog, 0.15, 0.002, moderngl.LINES)

    def render_glow_layer(self, vao, prog, base_glow, blur, mode):
        prog['u_blur_offset'].value = (0,0)
        prog['u_glow'].value = base_glow * 2.5
        vao.render(mode)
        prog['u_glow'].value = base_glow * 0.4
        for off in [(blur,0), (-blur,0), (0,blur), (0,-blur)]:
            prog['u_blur_offset'].value = off
            vao.render(mode)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: self.obj_pos[0] -= 0.1
        if keys[pygame.K_RIGHT]: self.obj_pos[0] += 0.1
        if keys[pygame.K_UP]: self.obj_pos[1] += 0.1
        if keys[pygame.K_DOWN]: self.obj_pos[1] -= 0.1
        if keys[pygame.K_w]: self.obj_pos[2] += 0.1
        if keys[pygame.K_s]: self.obj_pos[2] -= 0.1

        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and not self.is_recording: self.start_gif_render()
                if event.key == pygame.K_SPACE:
                    self.obj_index = (self.obj_index + 1) % len(self.obj_files) if self.obj_files else 0
                    self.load_current_model()
                if event.key == pygame.K_v: self.auto_rotate = not self.auto_rotate
            if event.type == pygame.MOUSEWHEEL: self.model_scale = max(1.0, self.model_scale + event.y)
            if event.type == pygame.MOUSEBUTTONDOWN: self.dragging = True
            if event.type == pygame.MOUSEBUTTONUP: self.dragging = False
            if event.type == pygame.MOUSEMOTION and self.dragging:
                self.mouse_rot[1] += event.rel[0] * 0.01
                self.mouse_rot[0] += event.rel[1] * 0.01
        return True

    def start_gif_render(self):
        self.is_recording = True
        fps = 30
        total_frames = int(self.loop_duration * fps)
        video_filename = "ultra_hq_temp.mp4"
        final_output = "telegram_hq_loop.mp4"
        
        print(f"Запуск HQ рендеринга (M4 Optimized)... Кадров: {total_frames}")
        
        # Используем imageio для записи "сырого" потока, но финальную жмем через ffmpeg вручную
        writer = imageio.get_writer(
            video_filename,
            fps=fps,
            codec='libx264',
            # CRF 17 - это практически визуальный оригинал.
            # Preset veryslow - лучший анализ векторов движения для плавности
            output_params=[
                '-crf', '17',
                '-preset', 'veryslow',
                '-tune', 'animation',
                '-pix_fmt', 'yuv420p'
            ]
        )

        for i in range(total_frames):
            pygame.event.pump()
            t = (i / total_frames) * self.loop_duration
            self.render_scene(t)
            
            raw_data = self.ctx.screen.read(components=3)
            img = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.res[1], self.res[0], 3))
            writer.append_data(np.flipud(img))
            
            if i % 10 == 0:
                print(f"Кадр {i}/{total_frames} ({int(i/total_frames*100)}%)")

        writer.close()
        
        # Финальный проход через ffmpeg для Telegram-флагов без потери качества
        print("Финализация контейнера...")
        cmd = [
            'ffmpeg', '-y',
            '-i', video_filename,
            '-c', 'copy', # Просто копируем уже идеально пожатый поток в финальный контейнер
            '-movflags', 'faststart',
            final_output
        ]
        subprocess.run(cmd)
        
        if os.path.exists(video_filename):
            os.remove(video_filename)
            
        print(f"Готово! Файл сохранен: {final_output}")
        self.is_recording = False

    def run(self):
        clock = pygame.time.Clock()
        while self.handle_input():
            if not self.is_recording:
                t = time.time() - self.start_t
                self.render_scene(t)
                pygame.display.flip()
            clock.tick(60)

if __name__ == "__main__":
    HQGifMaker().run()
