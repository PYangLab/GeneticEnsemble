/**
 * code adapted from Sunny Locus
 * 
 */

package au.edu.usyd.it.yangpy.snp;

import java.util.LinkedList;

public class ThreadPool extends ThreadGroup {  
    private boolean isClosed = false;  // whether the thread pool is closed
    private LinkedList workQueue;      // work queue 
    private static int threadPoolID = 1;  // thread pool id  
    
    public ThreadPool(int poolSize) {  // pool size
  
        super(threadPoolID + "");      //
        setDaemon(true);               // 
        workQueue = new LinkedList();  // create work queue
        for(int i = 0; i < poolSize; i++) {  
            new WorkThread(i).start();   // create and initiate a work threads according the pool size
        }  
    }  
      
    /** adding tasks to the work queue */  
    public synchronized void execute(Runnable task) {  
        if(isClosed) {  
            throw new IllegalStateException();  
        }  
        if(task != null) {  
            workQueue.add(task);// add a task
            notify();           //
        }  
    }  
      
    /** */  
    private synchronized Runnable getTask(int threadid) throws InterruptedException {  
        while(workQueue.size() == 0) {  
            if(isClosed) return null;  
            //System.out.println("work thread "+threadid+" wait for task...");  
            wait();             // if there is no ask, wait for it.
        }  
        //System.out.println("work thread "+threadid+" working on the task...");  
        return (Runnable) workQueue.removeFirst(); // return the first element and remove it from the work queue  
    }  
      
    /** close the thread pool */  
    public synchronized void closePool() {  
        if(! isClosed) {  
            waitFinish();        // wait for all task to be finished  
            isClosed = true;  
            workQueue.clear();  // clean the queue
            interrupt();        // terminate all thread in the thread pool
        }  
    }  
      
    /** wait for work threads to finish all the tasks */  
    public void waitFinish() {  
        synchronized (this) {  
            isClosed = true;  
            notifyAll();            //
        }  
        Thread[] threads = new Thread[activeCount()]; //activeCount() 
        int count = enumerate(threads); //enumerate()
        for(int i =0; i < count; i++) { //
            try {  
                threads[i].join();  //
            }catch(InterruptedException ex) {  
                ex.printStackTrace();  
            }  
        }  
    }  
  
    /** 
     */  
    private class WorkThread extends Thread {  
        private int id;  
        public WorkThread(int id) {  
            //
            super(ThreadPool.this,id+"");  
            this.id = id;  
        }  
        public void run() {  
            while(! isInterrupted()) {  //
                Runnable task = null;  
                try {  
                    task = getTask(id);     //
                }catch(InterruptedException ex) {  
                    ex.printStackTrace();  
                }  
                //
                if(task == null) return;  
                  
                try {  
                    task.run();  //
                }catch(Throwable t) {  
                    t.printStackTrace();  
                }  
            }//  end while  
        }//  end run  
    }// end workThread  
} 
