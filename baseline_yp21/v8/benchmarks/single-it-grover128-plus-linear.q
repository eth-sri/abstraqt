// Grover search on 128 qubits and 128 helper qubits

circuit: 256 qubits

// Init
H(0)
H(1)
H(2)
H(3)
H(4)
H(5)
H(6)
H(7)
H(8)
H(9)
H(10)
H(11)
H(12)
H(13)
H(14)
H(15)
H(16)
H(17)
H(18)
H(19)
H(20)
H(21)
H(22)
H(23)
H(24)
H(25)
H(26)
H(27)
H(28)
H(29)
H(30)
H(31)
H(32)
H(33)
H(34)
H(35)
H(36)
H(37)
H(38)
H(39)
H(40)
H(41)
H(42)
H(43)
H(44)
H(45)
H(46)
H(47)
H(48)
H(49)
H(50)
H(51)
H(52)
H(53)
H(54)
H(55)
H(56)
H(57)
H(58)
H(59)
H(60)
H(61)
H(62)
H(63)
H(64)
H(65)
H(66)
H(67)
H(68)
H(69)
H(70)
H(71)
H(72)
H(73)
H(74)
H(75)
H(76)
H(77)
H(78)
H(79)
H(80)
H(81)
H(82)
H(83)
H(84)
H(85)
H(86)
H(87)
H(88)
H(89)
H(90)
H(91)
H(92)
H(93)
H(94)
H(95)
H(96)
H(97)
H(98)
H(99)
H(100)
H(101)
H(102)
H(103)
H(104)
H(105)
H(106)
H(107)
H(108)
H(109)
H(110)
H(111)
H(112)
H(113)
H(114)
H(115)
H(116)
H(117)
H(118)
H(119)
H(120)
H(121)
H(122)
H(123)
H(124)
H(125)
H(126)
H(127)

// Grover iteration 1
NCNCNOT(0,1,128)
NCCNOT(2,128,129)
NCCNOT(3,129,130)
NCCNOT(4,130,131)
NCCNOT(5,131,132)
NCCNOT(6,132,133)
NCCNOT(7,133,134)
NCCNOT(8,134,135)
NCCNOT(9,135,136)
NCCNOT(10,136,137)
NCCNOT(11,137,138)
NCCNOT(12,138,139)
NCCNOT(13,139,140)
NCCNOT(14,140,141)
NCCNOT(15,141,142)
NCCNOT(16,142,143)
NCCNOT(17,143,144)
NCCNOT(18,144,145)
NCCNOT(19,145,146)
NCCNOT(20,146,147)
NCCNOT(21,147,148)
NCCNOT(22,148,149)
NCCNOT(23,149,150)
NCCNOT(24,150,151)
NCCNOT(25,151,152)
NCCNOT(26,152,153)
NCCNOT(27,153,154)
NCCNOT(28,154,155)
NCCNOT(29,155,156)
NCCNOT(30,156,157)
NCCNOT(31,157,158)
NCCNOT(32,158,159)
NCCNOT(33,159,160)
NCCNOT(34,160,161)
NCCNOT(35,161,162)
NCCNOT(36,162,163)
NCCNOT(37,163,164)
NCCNOT(38,164,165)
NCCNOT(39,165,166)
NCCNOT(40,166,167)
NCCNOT(41,167,168)
NCCNOT(42,168,169)
NCCNOT(43,169,170)
NCCNOT(44,170,171)
NCCNOT(45,171,172)
NCCNOT(46,172,173)
NCCNOT(47,173,174)
NCCNOT(48,174,175)
NCCNOT(49,175,176)
NCCNOT(50,176,177)
NCCNOT(51,177,178)
NCCNOT(52,178,179)
NCCNOT(53,179,180)
NCCNOT(54,180,181)
NCCNOT(55,181,182)
NCCNOT(56,182,183)
NCCNOT(57,183,184)
NCCNOT(58,184,185)
NCCNOT(59,185,186)
NCCNOT(60,186,187)
NCCNOT(61,187,188)
NCCNOT(62,188,189)
NCCNOT(63,189,190)
NCCNOT(64,190,191)
NCCNOT(65,191,192)
NCCNOT(66,192,193)
NCCNOT(67,193,194)
NCCNOT(68,194,195)
NCCNOT(69,195,196)
NCCNOT(70,196,197)
NCCNOT(71,197,198)
NCCNOT(72,198,199)
NCCNOT(73,199,200)
NCCNOT(74,200,201)
NCCNOT(75,201,202)
NCCNOT(76,202,203)
NCCNOT(77,203,204)
NCCNOT(78,204,205)
NCCNOT(79,205,206)
NCCNOT(80,206,207)
NCCNOT(81,207,208)
NCCNOT(82,208,209)
NCCNOT(83,209,210)
NCCNOT(84,210,211)
NCCNOT(85,211,212)
NCCNOT(86,212,213)
NCCNOT(87,213,214)
NCCNOT(88,214,215)
NCCNOT(89,215,216)
NCCNOT(90,216,217)
NCCNOT(91,217,218)
NCCNOT(92,218,219)
NCCNOT(93,219,220)
NCCNOT(94,220,221)
NCCNOT(95,221,222)
NCCNOT(96,222,223)
NCCNOT(97,223,224)
NCCNOT(98,224,225)
NCCNOT(99,225,226)
NCCNOT(100,226,227)
NCCNOT(101,227,228)
NCCNOT(102,228,229)
NCCNOT(103,229,230)
NCCNOT(104,230,231)
NCCNOT(105,231,232)
NCCNOT(106,232,233)
NCCNOT(107,233,234)
NCCNOT(108,234,235)
NCCNOT(109,235,236)
NCCNOT(110,236,237)
NCCNOT(111,237,238)
NCCNOT(112,238,239)
NCCNOT(113,239,240)
NCCNOT(114,240,241)
NCCNOT(115,241,242)
NCCNOT(116,242,243)
NCCNOT(117,243,244)
NCCNOT(118,244,245)
NCCNOT(119,245,246)
NCCNOT(120,246,247)
NCCNOT(121,247,248)
NCCNOT(122,248,249)
NCCNOT(123,249,250)
NCCNOT(124,250,251)
NCCNOT(125,251,252)
NCCNOT(126,252,253)
NCCNOT(127,253,254)
CNOT(254,255)
// The key step!
Z(255)
CNOT(254,255)
NCCNOT(127,253,254)
NCCNOT(126,252,253)
NCCNOT(125,251,252)
NCCNOT(124,250,251)
NCCNOT(123,249,250)
NCCNOT(122,248,249)
NCCNOT(121,247,248)
NCCNOT(120,246,247)
NCCNOT(119,245,246)
NCCNOT(118,244,245)
NCCNOT(117,243,244)
NCCNOT(116,242,243)
NCCNOT(115,241,242)
NCCNOT(114,240,241)
NCCNOT(113,239,240)
NCCNOT(112,238,239)
NCCNOT(111,237,238)
NCCNOT(110,236,237)
NCCNOT(109,235,236)
NCCNOT(108,234,235)
NCCNOT(107,233,234)
NCCNOT(106,232,233)
NCCNOT(105,231,232)
NCCNOT(104,230,231)
NCCNOT(103,229,230)
NCCNOT(102,228,229)
NCCNOT(101,227,228)
NCCNOT(100,226,227)
NCCNOT(99,225,226)
NCCNOT(98,224,225)
NCCNOT(97,223,224)
NCCNOT(96,222,223)
NCCNOT(95,221,222)
NCCNOT(94,220,221)
NCCNOT(93,219,220)
NCCNOT(92,218,219)
NCCNOT(91,217,218)
NCCNOT(90,216,217)
NCCNOT(89,215,216)
NCCNOT(88,214,215)
NCCNOT(87,213,214)
NCCNOT(86,212,213)
NCCNOT(85,211,212)
NCCNOT(84,210,211)
NCCNOT(83,209,210)
NCCNOT(82,208,209)
NCCNOT(81,207,208)
NCCNOT(80,206,207)
NCCNOT(79,205,206)
NCCNOT(78,204,205)
NCCNOT(77,203,204)
NCCNOT(76,202,203)
NCCNOT(75,201,202)
NCCNOT(74,200,201)
NCCNOT(73,199,200)
NCCNOT(72,198,199)
NCCNOT(71,197,198)
NCCNOT(70,196,197)
NCCNOT(69,195,196)
NCCNOT(68,194,195)
NCCNOT(67,193,194)
NCCNOT(66,192,193)
NCCNOT(65,191,192)
NCCNOT(64,190,191)
NCCNOT(63,189,190)
NCCNOT(62,188,189)
NCCNOT(61,187,188)
NCCNOT(60,186,187)
NCCNOT(59,185,186)
NCCNOT(58,184,185)
NCCNOT(57,183,184)
NCCNOT(56,182,183)
NCCNOT(55,181,182)
NCCNOT(54,180,181)
NCCNOT(53,179,180)
NCCNOT(52,178,179)
NCCNOT(51,177,178)
NCCNOT(50,176,177)
NCCNOT(49,175,176)
NCCNOT(48,174,175)
NCCNOT(47,173,174)
NCCNOT(46,172,173)
NCCNOT(45,171,172)
NCCNOT(44,170,171)
NCCNOT(43,169,170)
NCCNOT(42,168,169)
NCCNOT(41,167,168)
NCCNOT(40,166,167)
NCCNOT(39,165,166)
NCCNOT(38,164,165)
NCCNOT(37,163,164)
NCCNOT(36,162,163)
NCCNOT(35,161,162)
NCCNOT(34,160,161)
NCCNOT(33,159,160)
NCCNOT(32,158,159)
NCCNOT(31,157,158)
NCCNOT(30,156,157)
NCCNOT(29,155,156)
NCCNOT(28,154,155)
NCCNOT(27,153,154)
NCCNOT(26,152,153)
NCCNOT(25,151,152)
NCCNOT(24,150,151)
NCCNOT(23,149,150)
NCCNOT(22,148,149)
NCCNOT(21,147,148)
NCCNOT(20,146,147)
NCCNOT(19,145,146)
NCCNOT(18,144,145)
NCCNOT(17,143,144)
NCCNOT(16,142,143)
NCCNOT(15,141,142)
NCCNOT(14,140,141)
NCCNOT(13,139,140)
NCCNOT(12,138,139)
NCCNOT(11,137,138)
NCCNOT(10,136,137)
NCCNOT(9,135,136)
NCCNOT(8,134,135)
NCCNOT(7,133,134)
NCCNOT(6,132,133)
NCCNOT(5,131,132)
NCCNOT(4,130,131)
NCCNOT(3,129,130)
NCCNOT(2,128,129)
NCNCNOT(0,1,128)

H(0)
H(1)
H(2)
H(3)
H(4)
H(5)
H(6)
H(7)
H(8)
H(9)
H(10)
H(11)
H(12)
H(13)
H(14)
H(15)
H(16)
H(17)
H(18)
H(19)
H(20)
H(21)
H(22)
H(23)
H(24)
H(25)
H(26)
H(27)
H(28)
H(29)
H(30)
H(31)
H(32)
H(33)
H(34)
H(35)
H(36)
H(37)
H(38)
H(39)
H(40)
H(41)
H(42)
H(43)
H(44)
H(45)
H(46)
H(47)
H(48)
H(49)
H(50)
H(51)
H(52)
H(53)
H(54)
H(55)
H(56)
H(57)
H(58)
H(59)
H(60)
H(61)
H(62)
H(63)
H(64)
H(65)
H(66)
H(67)
H(68)
H(69)
H(70)
H(71)
H(72)
H(73)
H(74)
H(75)
H(76)
H(77)
H(78)
H(79)
H(80)
H(81)
H(82)
H(83)
H(84)
H(85)
H(86)
H(87)
H(88)
H(89)
H(90)
H(91)
H(92)
H(93)
H(94)
H(95)
H(96)
H(97)
H(98)
H(99)
H(100)
H(101)
H(102)
H(103)
H(104)
H(105)
H(106)
H(107)
H(108)
H(109)
H(110)
H(111)
H(112)
H(113)
H(114)
H(115)
H(116)
H(117)
H(118)
H(119)
H(120)
H(121)
H(122)
H(123)
H(124)
H(125)
H(126)
H(127)

NCNCNOT(0,1,128)
NCCNOT(2,128,129)
NCCNOT(3,129,130)
NCCNOT(4,130,131)
NCCNOT(5,131,132)
NCCNOT(6,132,133)
NCCNOT(7,133,134)
NCCNOT(8,134,135)
NCCNOT(9,135,136)
NCCNOT(10,136,137)
NCCNOT(11,137,138)
NCCNOT(12,138,139)
NCCNOT(13,139,140)
NCCNOT(14,140,141)
NCCNOT(15,141,142)
NCCNOT(16,142,143)
NCCNOT(17,143,144)
NCCNOT(18,144,145)
NCCNOT(19,145,146)
NCCNOT(20,146,147)
NCCNOT(21,147,148)
NCCNOT(22,148,149)
NCCNOT(23,149,150)
NCCNOT(24,150,151)
NCCNOT(25,151,152)
NCCNOT(26,152,153)
NCCNOT(27,153,154)
NCCNOT(28,154,155)
NCCNOT(29,155,156)
NCCNOT(30,156,157)
NCCNOT(31,157,158)
NCCNOT(32,158,159)
NCCNOT(33,159,160)
NCCNOT(34,160,161)
NCCNOT(35,161,162)
NCCNOT(36,162,163)
NCCNOT(37,163,164)
NCCNOT(38,164,165)
NCCNOT(39,165,166)
NCCNOT(40,166,167)
NCCNOT(41,167,168)
NCCNOT(42,168,169)
NCCNOT(43,169,170)
NCCNOT(44,170,171)
NCCNOT(45,171,172)
NCCNOT(46,172,173)
NCCNOT(47,173,174)
NCCNOT(48,174,175)
NCCNOT(49,175,176)
NCCNOT(50,176,177)
NCCNOT(51,177,178)
NCCNOT(52,178,179)
NCCNOT(53,179,180)
NCCNOT(54,180,181)
NCCNOT(55,181,182)
NCCNOT(56,182,183)
NCCNOT(57,183,184)
NCCNOT(58,184,185)
NCCNOT(59,185,186)
NCCNOT(60,186,187)
NCCNOT(61,187,188)
NCCNOT(62,188,189)
NCCNOT(63,189,190)
NCCNOT(64,190,191)
NCCNOT(65,191,192)
NCCNOT(66,192,193)
NCCNOT(67,193,194)
NCCNOT(68,194,195)
NCCNOT(69,195,196)
NCCNOT(70,196,197)
NCCNOT(71,197,198)
NCCNOT(72,198,199)
NCCNOT(73,199,200)
NCCNOT(74,200,201)
NCCNOT(75,201,202)
NCCNOT(76,202,203)
NCCNOT(77,203,204)
NCCNOT(78,204,205)
NCCNOT(79,205,206)
NCCNOT(80,206,207)
NCCNOT(81,207,208)
NCCNOT(82,208,209)
NCCNOT(83,209,210)
NCCNOT(84,210,211)
NCCNOT(85,211,212)
NCCNOT(86,212,213)
NCCNOT(87,213,214)
NCCNOT(88,214,215)
NCCNOT(89,215,216)
NCCNOT(90,216,217)
NCCNOT(91,217,218)
NCCNOT(92,218,219)
NCCNOT(93,219,220)
NCCNOT(94,220,221)
NCCNOT(95,221,222)
NCCNOT(96,222,223)
NCCNOT(97,223,224)
NCCNOT(98,224,225)
NCCNOT(99,225,226)
NCCNOT(100,226,227)
NCCNOT(101,227,228)
NCCNOT(102,228,229)
NCCNOT(103,229,230)
NCCNOT(104,230,231)
NCCNOT(105,231,232)
NCCNOT(106,232,233)
NCCNOT(107,233,234)
NCCNOT(108,234,235)
NCCNOT(109,235,236)
NCCNOT(110,236,237)
NCCNOT(111,237,238)
NCCNOT(112,238,239)
NCCNOT(113,239,240)
NCCNOT(114,240,241)
NCCNOT(115,241,242)
NCCNOT(116,242,243)
NCCNOT(117,243,244)
NCCNOT(118,244,245)
NCCNOT(119,245,246)
NCCNOT(120,246,247)
NCCNOT(121,247,248)
NCCNOT(122,248,249)
NCCNOT(123,249,250)
NCCNOT(124,250,251)
NCCNOT(125,251,252)
NCCNOT(126,252,253)
NCCNOT(127,253,254)
CNOT(254,255)
// The key step!
Z(255)
CNOT(254,255)
NCCNOT(127,253,254)
NCCNOT(126,252,253)
NCCNOT(125,251,252)
NCCNOT(124,250,251)
NCCNOT(123,249,250)
NCCNOT(122,248,249)
NCCNOT(121,247,248)
NCCNOT(120,246,247)
NCCNOT(119,245,246)
NCCNOT(118,244,245)
NCCNOT(117,243,244)
NCCNOT(116,242,243)
NCCNOT(115,241,242)
NCCNOT(114,240,241)
NCCNOT(113,239,240)
NCCNOT(112,238,239)
NCCNOT(111,237,238)
NCCNOT(110,236,237)
NCCNOT(109,235,236)
NCCNOT(108,234,235)
NCCNOT(107,233,234)
NCCNOT(106,232,233)
NCCNOT(105,231,232)
NCCNOT(104,230,231)
NCCNOT(103,229,230)
NCCNOT(102,228,229)
NCCNOT(101,227,228)
NCCNOT(100,226,227)
NCCNOT(99,225,226)
NCCNOT(98,224,225)
NCCNOT(97,223,224)
NCCNOT(96,222,223)
NCCNOT(95,221,222)
NCCNOT(94,220,221)
NCCNOT(93,219,220)
NCCNOT(92,218,219)
NCCNOT(91,217,218)
NCCNOT(90,216,217)
NCCNOT(89,215,216)
NCCNOT(88,214,215)
NCCNOT(87,213,214)
NCCNOT(86,212,213)
NCCNOT(85,211,212)
NCCNOT(84,210,211)
NCCNOT(83,209,210)
NCCNOT(82,208,209)
NCCNOT(81,207,208)
NCCNOT(80,206,207)
NCCNOT(79,205,206)
NCCNOT(78,204,205)
NCCNOT(77,203,204)
NCCNOT(76,202,203)
NCCNOT(75,201,202)
NCCNOT(74,200,201)
NCCNOT(73,199,200)
NCCNOT(72,198,199)
NCCNOT(71,197,198)
NCCNOT(70,196,197)
NCCNOT(69,195,196)
NCCNOT(68,194,195)
NCCNOT(67,193,194)
NCCNOT(66,192,193)
NCCNOT(65,191,192)
NCCNOT(64,190,191)
NCCNOT(63,189,190)
NCCNOT(62,188,189)
NCCNOT(61,187,188)
NCCNOT(60,186,187)
NCCNOT(59,185,186)
NCCNOT(58,184,185)
NCCNOT(57,183,184)
NCCNOT(56,182,183)
NCCNOT(55,181,182)
NCCNOT(54,180,181)
NCCNOT(53,179,180)
NCCNOT(52,178,179)
NCCNOT(51,177,178)
NCCNOT(50,176,177)
NCCNOT(49,175,176)
NCCNOT(48,174,175)
NCCNOT(47,173,174)
NCCNOT(46,172,173)
NCCNOT(45,171,172)
NCCNOT(44,170,171)
NCCNOT(43,169,170)
NCCNOT(42,168,169)
NCCNOT(41,167,168)
NCCNOT(40,166,167)
NCCNOT(39,165,166)
NCCNOT(38,164,165)
NCCNOT(37,163,164)
NCCNOT(36,162,163)
NCCNOT(35,161,162)
NCCNOT(34,160,161)
NCCNOT(33,159,160)
NCCNOT(32,158,159)
NCCNOT(31,157,158)
NCCNOT(30,156,157)
NCCNOT(29,155,156)
NCCNOT(28,154,155)
NCCNOT(27,153,154)
NCCNOT(26,152,153)
NCCNOT(25,151,152)
NCCNOT(24,150,151)
NCCNOT(23,149,150)
NCCNOT(22,148,149)
NCCNOT(21,147,148)
NCCNOT(20,146,147)
NCCNOT(19,145,146)
NCCNOT(18,144,145)
NCCNOT(17,143,144)
NCCNOT(16,142,143)
NCCNOT(15,141,142)
NCCNOT(14,140,141)
NCCNOT(13,139,140)
NCCNOT(12,138,139)
NCCNOT(11,137,138)
NCCNOT(10,136,137)
NCCNOT(9,135,136)
NCCNOT(8,134,135)
NCCNOT(7,133,134)
NCCNOT(6,132,133)
NCCNOT(5,131,132)
NCCNOT(4,130,131)
NCCNOT(3,129,130)
NCCNOT(2,128,129)
NCNCNOT(0,1,128)

H(0)
H(1)
H(2)
H(3)
H(4)
H(5)
H(6)
H(7)
H(8)
H(9)
H(10)
H(11)
H(12)
H(13)
H(14)
H(15)
H(16)
H(17)
H(18)
H(19)
H(20)
H(21)
H(22)
H(23)
H(24)
H(25)
H(26)
H(27)
H(28)
H(29)
H(30)
H(31)
H(32)
H(33)
H(34)
H(35)
H(36)
H(37)
H(38)
H(39)
H(40)
H(41)
H(42)
H(43)
H(44)
H(45)
H(46)
H(47)
H(48)
H(49)
H(50)
H(51)
H(52)
H(53)
H(54)
H(55)
H(56)
H(57)
H(58)
H(59)
H(60)
H(61)
H(62)
H(63)
H(64)
H(65)
H(66)
H(67)
H(68)
H(69)
H(70)
H(71)
H(72)
H(73)
H(74)
H(75)
H(76)
H(77)
H(78)
H(79)
H(80)
H(81)
H(82)
H(83)
H(84)
H(85)
H(86)
H(87)
H(88)
H(89)
H(90)
H(91)
H(92)
H(93)
H(94)
H(95)
H(96)
H(97)
H(98)
H(99)
H(100)
H(101)
H(102)
H(103)
H(104)
H(105)
H(106)
H(107)
H(108)
H(109)
H(110)
H(111)
H(112)
H(113)
H(114)
H(115)
H(116)
H(117)
H(118)
H(119)
H(120)
H(121)
H(122)
H(123)
H(124)
H(125)
H(126)
H(127)

assert state in span {
  |0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000> ,
  |00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
}

measure 0..255
