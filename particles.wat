(module
  ;; Memory: 1 page (64KB) for particle data
  ;; Layout: [x0, y0, vx0, vy0, x1, y1, vx1, vy1, ...]
  ;; Each particle: 4 floats = 16 bytes
  (memory (export "memory") 1)
  
  ;; Constants
  (global $G f32 (f32.const 0.5))        ;; Gravity constant
  (global $DT f32 (f32.const 0.016))     ;; Time step (~60fps)
  (global $SOFTENING f32 (f32.const 5.0)) ;; Prevents division by zero
  
  ;; Get particle X position (particle index -> memory offset)
  (func $get_x (param $i i32) (result f32)
    (f32.load (i32.mul (local.get $i) (i32.const 16)))
  )
  
  (func $get_y (param $i i32) (result f32)
    (f32.load (i32.add (i32.mul (local.get $i) (i32.const 16)) (i32.const 4)))
  )
  
  (func $get_vx (param $i i32) (result f32)
    (f32.load (i32.add (i32.mul (local.get $i) (i32.const 16)) (i32.const 8)))
  )
  
  (func $get_vy (param $i i32) (result f32)
    (f32.load (i32.add (i32.mul (local.get $i) (i32.const 16)) (i32.const 12)))
  )
  
  ;; Set particle properties
  (func $set_x (param $i i32) (param $val f32)
    (f32.store (i32.mul (local.get $i) (i32.const 16)) (local.get $val))
  )
  
  (func $set_y (param $i i32) (param $val f32)
    (f32.store (i32.add (i32.mul (local.get $i) (i32.const 16)) (i32.const 4)) (local.get $val))
  )
  
  (func $set_vx (param $i i32) (param $val f32)
    (f32.store (i32.add (i32.mul (local.get $i) (i32.const 16)) (i32.const 8)) (local.get $val))
  )
  
  (func $set_vy (param $i i32) (param $val f32)
    (f32.store (i32.add (i32.mul (local.get $i) (i32.const 16)) (i32.const 12)) (local.get $val))
  )
  
  ;; Initialize particle at index
  (func (export "init_particle") (param $i i32) (param $x f32) (param $y f32) (param $vx f32) (param $vy f32)
    (call $set_x (local.get $i) (local.get $x))
    (call $set_y (local.get $i) (local.get $y))
    (call $set_vx (local.get $i) (local.get $vx))
    (call $set_vy (local.get $i) (local.get $vy))
  )
  
  ;; Update all particles with N-body gravity
  (func (export "update") (param $n i32)
    (local $i i32)
    (local $j i32)
    (local $xi f32) (local $yi f32)
    (local $xj f32) (local $yj f32)
    (local $dx f32) (local $dy f32)
    (local $dist_sq f32) (local $dist f32)
    (local $force f32)
    (local $ax f32) (local $ay f32)
    (local $vx f32) (local $vy f32)
    
    ;; For each particle i
    (local.set $i (i32.const 0))
    (block $break_i
      (loop $loop_i
        (br_if $break_i (i32.ge_u (local.get $i) (local.get $n)))
        
        (local.set $xi (call $get_x (local.get $i)))
        (local.set $yi (call $get_y (local.get $i)))
        (local.set $ax (f32.const 0))
        (local.set $ay (f32.const 0))
        
        ;; Calculate force from all other particles j
        (local.set $j (i32.const 0))
        (block $break_j
          (loop $loop_j
            (br_if $break_j (i32.ge_u (local.get $j) (local.get $n)))
            
            (if (i32.ne (local.get $i) (local.get $j))
              (then
                (local.set $xj (call $get_x (local.get $j)))
                (local.set $yj (call $get_y (local.get $j)))
                
                (local.set $dx (f32.sub (local.get $xj) (local.get $xi)))
                (local.set $dy (f32.sub (local.get $yj) (local.get $yi)))
                
                ;; dist_sq = dx*dx + dy*dy + softening
                (local.set $dist_sq 
                  (f32.add
                    (f32.add
                      (f32.mul (local.get $dx) (local.get $dx))
                      (f32.mul (local.get $dy) (local.get $dy))
                    )
                    (global.get $SOFTENING)
                  )
                )
                
                (local.set $dist (f32.sqrt (local.get $dist_sq)))
                
                ;; force = G / dist^3
                (local.set $force 
                  (f32.div 
                    (global.get $G)
                    (f32.mul (local.get $dist_sq) (local.get $dist))
                  )
                )
                
                ;; accumulate acceleration
                (local.set $ax (f32.add (local.get $ax) (f32.mul (local.get $dx) (local.get $force))))
                (local.set $ay (f32.add (local.get $ay) (f32.mul (local.get $dy) (local.get $force))))
              )
            )
            
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $loop_j)
          )
        )
        
        ;; Update velocity: v += a * dt
        (local.set $vx (f32.add (call $get_vx (local.get $i)) (f32.mul (local.get $ax) (global.get $DT))))
        (local.set $vy (f32.add (call $get_vy (local.get $i)) (f32.mul (local.get $ay) (global.get $DT))))
        (call $set_vx (local.get $i) (local.get $vx))
        (call $set_vy (local.get $i) (local.get $vy))
        
        ;; Update position: x += v * dt
        (call $set_x (local.get $i) (f32.add (local.get $xi) (f32.mul (local.get $vx) (global.get $DT))))
        (call $set_y (local.get $i) (f32.add (local.get $yi) (f32.mul (local.get $vy) (global.get $DT))))
        
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop_i)
      )
    )
  )
  
  ;; Get particle data for rendering (returns ptr to memory)
  (func (export "get_particles_ptr") (result i32)
    (i32.const 0)
  )
)
