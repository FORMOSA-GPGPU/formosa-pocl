static int _errno_val = 0;
int *__errno(void) { return &_errno_val; }
